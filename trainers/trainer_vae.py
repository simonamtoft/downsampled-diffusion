import os
import wandb
import torch
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable

from .trainer import Trainer
from .train_helpers import DeterministicWarmup, \
    lambda_lr, bce_loss, log_images


class TrainerVAE(Trainer):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='tmp', mute:bool=True, res_folder:str='./results', n_channels:int=None):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute, res_folder, n_channels)
        
        # set latent sample dim
        if isinstance(self.config['z_dim'], list):
            self.sample_dim = self.config['z_dim'][0] 
        else:
            self.sample_dim = self.config['z_dim']

        # Define learning rate decay
        lr_decay = {
            'n_epochs': int(self.n_steps*2), 
            'delay': int(self.n_steps/2)
        }
        
        # Set optimizer and learning rate scheduler
        self.opt = Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=lambda_lr(**lr_decay)
        )
        
        # linear deterministic warmup over n epochs (from 0 to 1)
        self.gamma = DeterministicWarmup(n=100)
    
    def log_images(self, x_hat, epoch):
        # reshape reconstruction
        x_recon = x_hat[:self.n_samples]
        x_recon = torch.reshape(x_recon, (self.n_samples, self.n_channels, self.image_size, self.image_size))
        
        # sample from model
        x_mu = Variable(torch.randn(self.n_samples, self.sample_dim)).to(self.device)
        x_sample = self.model.sample(x_mu)
        x_sample = torch.reshape(x_sample, (self.n_samples, self.n_channels, self.image_size, self.image_size))

        # log recon and sample
        name = f'{epoch}_{self.name}_{self.config["dataset"]}'
        log_images(x_recon, x_sample, self.res_folder, name, self.n_rows)

    def train(self):
        # Instantiate wandb run
        wandb.init(project=self.wandb_name, config=self.config)
        wandb.watch(self.model)
        
        train_losses = []
        val_losses = []
        for epoch in range(self.n_steps):
            # Train Epoch
            self.model.train()
            elbo_train = []
            kld_train = []
            recon_train = []
            alpha = next(self.gamma)
            
            for x, _ in iter(self.train_loader):
                batch_size = x.size(0)

                # Pass batch through model
                x = x.view(batch_size, -1)
                x = Variable(x).to(self.device)
                x_hat, kld = self.model(x)
                
                # Compute losses
                recon = torch.mean(bce_loss(x_hat, x))
                kl = torch.mean(kld)
                loss = recon + alpha * kl
                elbo = -(recon + kl)
                
                # Update gradients
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
            
                # save losses
                elbo_train.append(torch.mean(elbo).item())
                kld_train.append(torch.mean(kl).item())
                recon_train.append(torch.mean(recon).item())
            
            # get mean losses
            recon_train = self.loss_handle(recon_train)
            kld_train = self.loss_handle(kld_train)
            elbo_train = self.loss_handle(elbo_train)
            
            # Log train losses
            train_losses.append(elbo_train)
            wandb.log({
                'recon_train': recon_train,
                'kl_train': kld_train,
                'loss_train': elbo_train
            }, commit=False)

            # Update scheduler
            self.scheduler.step()
        
            # Validation epoch
            self.model.eval()
            with torch.no_grad():
                elbo_val = []
                kld_val = []
                recon_val = []
                for x, _ in iter(self.val_loader):
                    batch_size = x.size(0)

                    # Pass batch through model
                    x = x.view(batch_size, -1)
                    x = Variable(x).to(self.device)
                    x_hat, kld = self.model(x)

                    # Compute losses
                    recon = torch.mean(bce_loss(x_hat, x))
                    kl = torch.mean(kld)
                    loss = recon + alpha * kl
                    elbo = -(recon + kl)

                    # save losses
                    elbo_val.append(torch.mean(elbo).item())
                    kld_val.append(torch.mean(kld).item())
                    recon_val.append(torch.mean(recon).item())
            
            # get mean losses
            recon_val = self.loss_handle(recon_val)
            kld_val = self.loss_handle(kld_val)
            elbo_val = self.loss_handle(elbo_val)

            # Log validation losses
            val_losses.append(elbo_val)
            wandb.log({
                'recon_val': recon_val,
                'kl_val': kld_val,
                'loss_val': elbo_val
            }, commit=False)
            
            # log images to wandb
            self.log_images(x_hat, epoch)

        # Finalize training
        self.finalize()
        return train_losses, val_losses
    