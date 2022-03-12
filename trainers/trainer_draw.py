import wandb
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from utils import min_max_norm_image
from .trainer import Trainer
from .train_helpers import DeterministicWarmup, \
    log_images, lambda_lr


class TrainerDRAW(Trainer):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='tmp', mute:bool=True, res_folder:str='./results', n_channels:int=None):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute, res_folder, n_channels)
        
        # Setup learning rate scheduler
        lr_decay = {'n_epochs': 1000, 'delay': 150}
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=lambda_lr(**lr_decay)
        )
        
        # linear deterministic warmup over n epochs (from 0 to 1)
        self.gamma = DeterministicWarmup(n=100)
        
        # Define loss function
        self.recon_fn = nn.BCELoss(reduction='none').to(self.device)

        # update name to include the depth of the model
        self.name += f'_{config["T"]}'

    def log_images(self, x_hat, epoch):
        log_shape = (self.n_samples, self.n_channels, self.image_size, self.image_size)
        
        # reshape reconstruction
        x_recon = x_hat[:self.n_samples]
        x_recon = torch.reshape(x_recon, log_shape)
        
        # sample from model
        x_sample = self.model.sample()
        x_sample = x_sample[:self.n_samples]
        x_sample = torch.reshape(x_sample, log_shape)
        
        # perform min-max normalization on samples
        x_sample = min_max_norm_image(x_sample)
    
        # log recon and sample
        name = f'{epoch}_{self.name}_{self.config["dataset"]}'
        log_images(x_recon, x_sample, self.res_folder, name, self.n_rows)
        
    def train(self):
        # Initialize a new wandb run
        wandb.init(project=self.wandb_name, config=self.config)
        wandb.watch(self.model)
        
        # Training and validation loop
        train_losses = []
        val_losses = []
        for epoch in range(self.n_steps):
            # Training Epoch
            self.model.train()
            loss_recon = []
            loss_kl = []
            loss_elbo = []
            alpha = next(self.gamma)
            for x, _ in iter(self.train_loader):
                # Pass through model
                x = x.view(self.batch_size, -1).to(self.device)
                x_hat, kld = self.model(x)
                x_hat = torch.sigmoid(x_hat)

                # Compute losses
                recon = self.recon_fn(x_hat, x).sum(1).mean()
                kl = kld.sum(1).mean()
                loss = recon + alpha * kl
                elbo = -(recon + kl)

                # Update gradients
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1)
                self.opt.step()
                self.opt.zero_grad()

                # save losses
                loss_recon.append(recon.item())
                loss_kl.append(kl.item())
                loss_elbo.append(elbo.item())
            
            # get mean losses
            loss_recon = self.loss_handle(loss_recon)
            loss_kl = self.loss_handle(loss_kl)
            loss_elbo = self.loss_handle(loss_elbo)

            # Log train stuff
            train_losses.append(loss_elbo)
            wandb.log({
                'recon_train': loss_recon,
                'kl_train': loss_kl,
                'loss_train': loss_elbo
            }, commit=False)

            # Update scheduler
            self.scheduler.step()

            # Evaluate on validation set
            if self.val_loader is not None:
                with torch.no_grad():
                    self.model.eval()
                    loss_recon = []
                    loss_kl = []
                    loss_elbo = []
                    for x, _ in iter(self.val_loader):
                        # Pass through model
                        x = x.view(self.batch_size, -1).to(self.device)
                        x_hat, kld = self.model(x)
                        x_hat = torch.sigmoid(x_hat)

                        # Compute losses
                        recon = self.recon_fn(x_hat, x).sum(1).mean()
                        kl = kld.sum(1).mean()
                        loss = recon + alpha * kl
                        elbo = -(recon + kl)

                        # save losses
                        loss_recon.append(recon.item())
                        loss_kl.append(kl.item())
                        loss_elbo.append(elbo.item())
                    
                    # get mean losses
                    loss_recon = self.loss_handle(loss_recon)
                    loss_kl = self.loss_handle(loss_kl)
                    loss_elbo = self.loss_handle(loss_elbo)

                    # Log validation losses
                    val_losses.append(loss_elbo)
                    wandb.log({
                        'recon_val': loss_recon,
                        'kl_val': loss_kl,
                        'loss_val': loss_elbo
                    }, commit=False)

                    # log images to wandb
                    self.log_images(x_hat, epoch)
        
        # Finalize training
        self.finalize()
        return train_losses, val_losses