import wandb
import torch
from torch import Tensor
import numpy as np

from utils import min_max_norm_image
from .ema import EMA
from .trainer import Trainer
from .train_helpers import cycle, log_images, \
    delete_if_exists


class TrainerDDPM(Trainer):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='tmp', mute:bool=True, res_folder:str='./results', n_channels:int=None):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute, res_folder, n_channels)
        # make data loaders cyclic
        self.train_loader = cycle(self.train_loader)
        self.val_loader = cycle(self.val_loader)
        
        # take a single image from the validation dataset
        # and turn it to batch
        self.val_batch = next(self.val_loader)[0][0] \
            .repeat(self.n_samples, 1, 1, 1) \
            .to(self.device)

        # instantiate training variables
        self.step = 0

        # specific DDPM trainer stuff
        self.gradient_accumulate_every = 2
        self.logging_every = 1000

        # EMA model for DDPM training
        self.use_ema = config['ema_decay'] > 0
        if self.use_ema:
            self.step_start_ema = 2000
            self.update_ema_every = 10
            self.ema = EMA(self.model, config['ema_decay'])
            self.ema.eval()
        
        # update name to include the depth of the model
        self.name += f'_{config["T"]}'

    def save_checkpoint(self) -> None:
        """Save the checkpoint of the training run locally and to wandb."""
        save_data = {
            'optimizer': self.opt.state_dict(),
            'model': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'step': self.step,
        }
        if self.use_ema:
            save_data['ema_model'] = self.ema.state_dict()
            
        torch.save(save_data, self.checkpoint_name)
        wandb.save(self.checkpoint_name, policy='live')
    
    def load_checkpoint(self, checkpoint:dict) -> None:
        """Load the state dict into the instantiated model and ema model."""
        self.opt.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['model'])
        self.config = checkpoint['config']
        self.train_losses = checkpoint['train_losses']
        self.step = checkpoint['step']
        if 'ema_model' in checkpoint:
            self.ema.load_state_dict(checkpoint['ema_model'])

    @torch.no_grad()
    def sample(self) -> Tensor:
        """Generate n_images samples from the EMA model."""
        if self.use_ema:
            return self.ema.sample(self.n_samples)
        else:
            return self.model.sample(self.n_samples)

    @torch.no_grad()
    def recon(self, x:Tensor) -> Tensor:
        """Generate n_images reconstructions from the model."""
        if self.use_ema:
            return self.ema.reconstruct(x, self.n_samples)
        else:
            return self.model.reconstruct(x, self.n_samples)

    @torch.no_grad()
    def log_wandb(self, x:Tensor, commit:bool=True) -> None:
        """Log reconstruction and sample images to wandb."""
        samples = min_max_norm_image(self.sample())
        recon = min_max_norm_image(self.recon(x))
        log_name = f'{self.step}_{self.name}_{self.config["dataset"]}'
        name_recon, name_sample = log_images(
            x_recon=recon,
            x_sample=samples,
            folder=self.res_folder,
            name=f'{log_name}',
            nrow=self.n_rows,
            commit=commit
        )
        delete_if_exists(name_recon)
        delete_if_exists(name_sample)

    def update_ema(self):
        if self.step < self.step_start_ema:
            self.ema.reset(self.model)
        elif self.step % self.update_ema_every == 0:
            self.ema.update(self.model)
    
    def train_loop(self) -> None:
        while self.step < self.n_steps:
            ####    TRAIN STEP   ####
            obj_list = []
            self.model.train()
            for _ in range(self.gradient_accumulate_every):
                # retrieve a training batch and port to device
                x, _ = next(self.train_loader)
                x = x.to(self.device)

                # perform a model forward pass
                obj = self.model(x)

                # compute gradients
                objective = obj / self.gradient_accumulate_every
                objective.backward()

                # save loss
                obj_list.append(obj.item())
            
            # store training loss
            train_obj = np.mean(obj_list)
            self.train_losses.append(train_obj)
            is_log = self.step != 0 and self.step % self.logging_every == 0
            wandb.log({
                'train_obj': train_obj
            }, commit=(not is_log))
            
            # update gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            self.opt.step()
            self.opt.zero_grad()

            # update EMA parameters
            if self.use_ema:
                self.update_ema()

            ####    EVAL STEP   ####
            self.model.eval()
            if is_log:
                self.save_checkpoint()
                self.log_wandb(self.val_batch)
                # delete_if_exists(self.checkpoint_name)

            ####    INCREMENT   ####
            self.step += 1


class TrainerDownsampleDDPM(TrainerDDPM):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='tmp', mute:bool=True, res_folder:str='./results', n_channels:int=None):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute, res_folder, n_channels)

    @torch.no_grad()
    def log_wandb(self, x:Tensor) -> None:
        """Log reconstruction and sample images, for both original and latent space to wandb."""
        # generate samples and reconstructions
        x_recon, z_recon = self.recon(x)
        x_sample, z_sample = self.sample()
        
        # convert latent samples and recon to single channel
        # done by meaning the channel dimension and set shape to be N x 1 x H x W
        z_recon = z_recon.mean(dim=1)[:, None]
        z_sample = z_sample.mean(dim=1)[:, None]

        # do min-max normalization
        x_recon, z_recon, x_sample, z_sample = (
            min_max_norm_image(x_recon), min_max_norm_image(z_recon), 
            min_max_norm_image(x_sample), min_max_norm_image(z_sample)
        )

        # define logging name
        log_name = f'{self.step}_{self.name}_{self.config["dataset"]}'
        
        # log original image space reconstructions and samples to wandb
        name_x_recon, name_x_sample = log_images(
            x_recon=x_recon,
            x_sample=x_sample,
            folder=self.res_folder,
            name=f'x_{log_name}',
            nrow=self.n_rows,
            commit=False
        )
        
        # log latent reconstructions and samples to wandb
        name_z_recon, name_z_sample = log_images(
            x_recon=z_recon,
            x_sample=z_sample,
            folder=self.res_folder,
            name=f'z_{log_name}',
            nrow=self.n_rows,
            rname='recon_latent',
            sname='sample_latent',
            commit=True
        )
        
        # delete images after logging
        delete_if_exists(name_x_recon)
        delete_if_exists(name_x_sample)
        delete_if_exists(name_z_recon)
        delete_if_exists(name_z_sample)

    def train_loop(self):
        while self.step < self.n_steps:
            ####    TRAIN STEP   ####
            self.model.train()
            obj_list, lat_list, rec_list = [], [], []
            for _ in range(self.gradient_accumulate_every):
                # retrieve a batch and port to device
                x, _ = next(self.train_loader)
                x = x.to(self.device)

                # perform a model forward pass
                obj, loss_dict = self.model(x)

                # backward pass
                objective = obj / self.gradient_accumulate_every
                objective.backward()

                # save losses
                obj_list.append(objective.item())
                lat_list.append(loss_dict['latent'].item())
                rec_list.append(loss_dict['recon'].item())

            # store training loss
            train_obj = np.mean(obj_list)
            train_latent = np.mean(lat_list)
            train_recon = np.mean(rec_list)
            self.train_losses.append(train_obj)
            is_log = self.step != 0 and self.step % self.logging_every == 0
            wandb.log({
                'train_obj': train_obj,
                'train_latent': train_latent,
                'train_recon': train_recon
            }, commit=(not is_log))
 
            # update gradients
            self.opt.step()
            self.opt.zero_grad()

            # update EMA parameters
            if self.use_ema:
                self.update_ema()

            ####    EVAL STEP   ####
            self.model.eval()
            if is_log:
                self.save_checkpoint()
                self.log_wandb(self.val_batch)
                # delete_if_exists(self.checkpoint_name)

            ####    INCREMENT   ####
            self.step += 1
