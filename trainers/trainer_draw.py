import os
import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .trainer import Trainer
from .train_helpers import DeterministicWarmup, \
    log_images, lambda_lr


class TrainerDRAW(Trainer):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='', mute:bool=True, n_channels:int=1):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute, n_channels=n_channels)
        
        # Setup learning rate scheduler
        lr_decay = {'n_epochs': 1000, 'delay': 150}
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=lambda_lr(**lr_decay)
        )
        
        # linear deterministic warmup over n epochs (from 0 to 1)
        self.gamma = DeterministicWarmup(n=100)
        
        # Define loss function
        if self.n_channels == 1:
            self.loss_fn = nn.BCELoss(reduction='none').to(self.device)
        else:
            # self.loss_fn = nn.MSELoss(reduction='none').to(self.device)
            self.loss_fn = nn.CrossEntropyLoss(reduction='none').to(self.device)

        # update name to include the depth of the model
        self.name += f'_{config["T"]}'

    def log_images(self, x_hat, epoch):
        # reshape reconstruction
        x_recon = x_hat[:self.n_samples]
        x_recon = torch.reshape(x_recon, (self.n_samples, self.n_channels, self.image_size, self.image_size))
        
        # sample from model
        x_sample = self.model.sample()
        x_sample = x_sample[:self.n_samples]
        x_sample = torch.reshape(x_sample, (self.n_samples, self.n_channels, self.image_size, self.image_size))
        
        print('recon:', x_recon.min(), x_recon.max())
        print('sample:', x_sample.min(), x_sample.max())

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
                batch_size = x.size(0)

                # Pass through model
                x = x.view(batch_size, -1).to(self.device)
                x_hat, kld = self.model(x)
                # x_hat = torch.sigmoid(x_hat)

                # Compute losses
                recon = torch.mean(self.loss_fn(x_hat, x.long()).sum(1))
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
            loss_recon = self.loss_handle(loss_recon, self.x_dim)
            loss_kl = self.loss_handle(loss_kl, self.x_dim)
            loss_elbo = self.loss_handle(loss_elbo, self.x_dim)

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
                    loss_recon = self.loss_handle(loss_recon, self.x_dim)
                    loss_kl = self.loss_handle(loss_kl, self.x_dim)
                    loss_elbo = self.loss_handle(loss_elbo, self.x_dim)

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
        save_path = f'{self.res_folder}/{self.name}_model.pt'
        self.save_to_wandb(save_path)
        wandb.finish()
        os.remove(save_path)
        print(f"Training of {self.name} completed!")
        return train_losses, val_losses