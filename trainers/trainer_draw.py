import json
import wandb
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_
from torch.distributions.normal import Normal

from .trainer import Trainer
from .train_helpers import DeterministicWarmup, log_images, \
    lambda_lr

class TrainerDRAW(Trainer):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='', mute:bool=True):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute)
        
        if config['dataset'] not in ['mnist', 'omniglot']:
            self.n_channels = 3
        else:
            self.n_channels = 1
        
        # Setup learning rate scheduler
        lr_decay = {'n_epochs': 1000, 'delay': 150}
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=lambda_lr(**lr_decay)
        )
        
        # linear deterministic warmup over n epochs (from 0 to 1)
        self.gamma = DeterministicWarmup(n=100)
        
        # Define loss function
        self.loss_fn = nn.BCELoss(reduction='none').to(self.device)
        
        # update name to include the depth of the model
        self.name += f'_{config["T"]}'
        
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
                batch_size = x.size(0)

                # Pass through model
                x = x.view(batch_size, -1).to(self.device)
                x_hat, kld = self.model(x)
                x_hat = torch.sigmoid(x_hat)

                # Compute losses
                recon = torch.mean(self.loss_fn(x_hat, x).sum(1))
                kl = torch.mean(kld.sum(1))
                loss = recon + alpha * kl
                elbo = -(recon + kl)

                # filter nan losses
                if not torch.isnan(loss):
                    # Update gradients
                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()

                # save losses
                loss_recon.append(recon.item())
                loss_kl.append(kl.item())
                loss_elbo.append(elbo.item())
            
            # get mean losses
            loss_recon = np.array(loss_recon).mean()
            loss_kl = np.array(loss_kl).mean()
            loss_elbo = np.array(loss_elbo).mean()

            # Log train stuff
            train_losses.append(loss_elbo)
            wandb.log({
                'recon_train': loss_recon,
                'kl_train': loss_kl,
                'loss_train': loss_elbo
            }, commit=False)

            # Update scheduler
            # if "lr_decay" in config:
            self.scheduler.step()

            # Evaluate on validation set
            if self.val_loader is not None:
                with torch.no_grad():
                    self.model.eval()
                    loss_recon = []
                    loss_kl = []
                    loss_elbo = []
                    for x, _ in iter(self.val_loader):
                        batch_size = x.size(0)

                        # Pass through model
                        x = x.view(batch_size, -1).to(self.device)
                        x_hat, kld = self.model(x)
                        x_hat = torch.sigmoid(x_hat)

                        # Compute losses
                        recon = torch.mean(self.loss_fn(x_hat, x).sum(1))
                        kl = torch.mean(kld.sum(1))
                        loss = recon + alpha * kl
                        elbo = -(recon + kl)

                        # save losses
                        loss_recon.append(recon.item())
                        loss_kl.append(kl.item())
                        loss_elbo.append(elbo.item())
                    
                    # get mean losses
                    loss_recon = np.array(loss_recon).mean()
                    loss_kl = np.array(loss_kl).mean()
                    loss_elbo = np.array(loss_elbo).mean()

                    # Log validation losses
                    val_losses.append(loss_elbo)
                    wandb.log({
                        'recon_val': loss_recon,
                        'kl_val': loss_kl,
                        'loss_val': loss_elbo
                    }, commit=False)

                    # Sample from model
                    x_sample = self.model.sample()
                    
                    print(x_sample.shape)

                    # Log images to wandb
                    log_images(x_hat, x_sample, f'{epoch}_{self.name}_{self.config["dataset"]}', self.res_folder, self.n_channels)
        
        # Finalize training
        self.save_to_wandb()
        # torch.save(self.model, f'{self.res_folder}/{self.name}_model.pt')
        # wandb.save(f'{self.res_folder}/{self.name}_model.pt')
        wandb.finish()
        return train_losses, val_losses