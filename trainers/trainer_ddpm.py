import os
import copy
import wandb
import torch

from .trainer import Trainer
from .train_helpers import cycle, num_to_groups, \
    log_images, min_max_norm, delete_if_exists


class EMA():
    def __init__(self, beta:float):
        """Exponential Moving Average of model parameters.
        
        Args:
            beta (float):    
        """
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        """
        Update the parameters of the EMA model with the new parameters of the model used in training.
        
        Args:
            ma_model (nn.module):   The EMA model 
        """
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old:torch.Tensor, new:torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class TrainerDDPM(Trainer):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='tmp', mute:bool=True, res_folder:str='./results', n_channels:int=None):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute, res_folder, n_channels)
        # set train loader as a cycle instead
        self.train_loader = cycle(self.train_loader)

        # initialize step count used for training
        self.step = 0
        
        # list of losses
        self.losses = []

        # specific DDPM trainer stuff
        self.gradient_accumulate_every = 2
        self.logging_every = 1000

        # EMA model for DDPM training...
        self.step_start_ema = 2000
        self.update_ema_every = 10
        self.ema = EMA(0.9999)
        self.ema_model = copy.deepcopy(self.model)
        self.reset_ema()

        # update name to include the depth of the model
        self.name += f'_{config["T"]}'
        self.checkpoint_name = os.path.join('./logging', f'checkpoint_{self.name}.pt')

        # define whether to log reconstructions and samples or not
        self.log_recon = True
        self.log_sample = True

    def reset_ema(self) -> None:
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self) -> None:
        if self.step < self.step_start_ema:
            self.reset_ema()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save_checkpoint(self) -> None:
        """Save the checkpoint of the training run locally and to wandb."""
        save_data = {
            'optimizer': self.opt.state_dict(),
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'config': self.config,
            'losses': self.losses,
            'step': self.step,
        }
        torch.save(save_data, self.checkpoint_name)
        wandb.save(self.checkpoint_name, policy='live')
    
    def load_checkpoint(self, checkpoint) -> None:
        """Load the state dict into the instantiated model and ema model."""
        self.opt.load_state_dict(checkpoint['optimizer'])
        self.model.load_state_dict(checkpoint['model'])
        self.ema_model.load_state_dict(checkpoint['ema_model'])
        self.config = checkpoint['config']
        self.losses = checkpoint['losses']
        self.step = checkpoint['step']

    @torch.no_grad()
    def sample(self) -> torch.Tensor:
        """Generate n_images samples from the EMA model."""
        batches = num_to_groups(self.n_samples, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
        samples = torch.cat(all_images_list, dim=0)
        return samples

    @torch.no_grad()
    def recon(self, x:torch.Tensor) -> torch.Tensor:
        """Generate n_images reconstructions from the model."""
        assert x.shape[0] >= self.n_samples
        x = x[:self.n_samples]
        x_recon = self.ema_model.reconstruct(x)
        return x_recon

    @torch.no_grad()
    def log_wandb(self, x:torch.Tensor) -> None:
        """Log reconstruction and sample images along with a training checkpoint to wandb."""
        # save model to wandb
        self.save_checkpoint()
        
        # generate samples and reconstructions
        samples = self.sample() if self.log_sample else None
        recon = self.recon(x) if self.log_recon else None
        
        # print min, max for samples and recon
        # perform minmax norm on both
        if self.log_sample:
            # print('sample:', samples.min(), samples.max())
            samples = min_max_norm(samples)
        if self.log_recon:
            # print('recon:', recon.min(), recon.max())
            recon = min_max_norm(recon)

        # define basename
        log_name = f'{self.step}_{self.name}_{self.config["dataset"]}'
        
        # get log dict etc.
        log_images(
            x_recon=recon, 
            x_sample=samples, 
            folder=self.res_folder, 
            name=f'{log_name}.png', 
            nrow=self.n_rows
        )

        # remove local checkpoint file
        delete_if_exists(self.checkpoint_name)
    
    def train_loop(self) -> list:
        while self.step < self.n_steps:
            train_obj = []
            self.model.train()
            for _ in range(self.gradient_accumulate_every):
                # retrieve a training batch and port to device
                x, _ = next(self.train_loader)
                x = x.to(self.device)

                # perform a model forward pass
                obj, _ = self.model(x)

                # compute gradients
                objective = obj / self.gradient_accumulate_every
                objective.backward()

                # save loss
                train_obj.append(obj.item())

            # update gradients
            self.opt.step()
            self.opt.zero_grad()

            # update EMA
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # eval stuff
            self.model.eval()
            loss_obj = self.loss_handle(train_obj)
            self.losses.append(loss_obj)
            is_log = self.step != 0 and self.step % self.logging_every == 0
            wandb.log({
                'train_obj': loss_obj
            }, commit=(not is_log))
            if is_log:
                self.log_wandb(x)

            # update step
            self.step += 1


class TrainerDownsampleDDPM(TrainerDDPM):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='tmp', mute:bool=True, res_folder:str='./results', n_channels:int=None):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute, res_folder, n_channels)

    def train_loop(self):
        while self.step < self.n_steps:
            train_obj = []
            train_latent = []
            train_recon = []
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
                train_obj.append(objective.item())
                train_latent.append(loss_dict['latent'].item())
                train_recon.append(loss_dict['recon'].item())

            # store losses
            loss_ = self.loss_handle(train_obj)
            loss_latent_ = self.loss_handle(train_latent)
            loss_recon_ = self.loss_handle(train_recon)
            self.losses.append(loss_)

            # update gradients
            self.opt.step()
            self.opt.zero_grad()

            # update EMA
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # log stuff to wandb
            is_log = self.step != 0 and self.step % self.logging_every == 0
            wandb.log({
                'train_obj': loss_,
                'train_latent': loss_latent_,
                'train_recon': loss_recon_
            }, commit=(not is_log))
            if is_log:
                self.log_wandb(x)

            # update step
            self.step += 1
