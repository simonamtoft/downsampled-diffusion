import wandb
import torch
import numpy as np
from torch.optim import Adam
from torch.autograd import Variable

from .trainer import Trainer
from .train_helpers import DeterministicWarmup, \
    lambda_lr, bce_loss, log_images, num_to_groups


class TrainerVAE(Trainer):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='', mute:bool=True):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute)
        
        # extract latent dim from config
        self.z_dim = self.config['z_dim']
        
        # Define learning rate decay
        lr_decay = {'n_epochs': 1000, 'delay': 150}
        
        # Set optimizer and learning rate scheduler
        self.opt = Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lr_lambda=lambda_lr(**lr_decay)
        )
        
        # linear deterministic warmup over n epochs (from 0 to 1)
        self.gamma = DeterministicWarmup(n=100)
    
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
            recon_train = np.array(recon_train).mean()
            kld_train = np.array(kld_train).mean()
            elbo_train = np.array(elbo_train).mean()
            
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
            recon_val = np.array(recon_val).mean()
            kld_val = np.array(kld_val).mean()
            elbo_val = np.array(elbo_val).mean()

            # Log validation losses
            val_losses.append(elbo_val)
            wandb.log({
                'recon_val': recon_val,
                'kl_val': kld_val,
                'loss_val': elbo_val
            }, commit=False)

            # Sample from model
            if isinstance(self.z_dim, list):
                x_mu = Variable(torch.randn(16, self.z_dim[0])).to(self.device)
            else:
                x_mu = Variable(torch.randn(16, self.z_dim)).to(self.device)
            x_sample = self.model.sample(x_mu)
            
            # Log images to wandb
            log_images(x_hat, x_sample, f'{epoch}_{self.name}_{self.config["dataset"]}', self.res_folder)
        
        # Finalize training
        torch.save(self.model, f'{self.res_folder}/{self.name}_model.pt')
        wandb.save(f'{self.res_folder}/{self.name}_model.pt')
        wandb.finish()
        print(f"Training of {self.name} completed!")
        return train_losses
    