import copy
import wandb
import torch
import numpy as np

from .trainer import Trainer
from .train_helpers import cycle, num_to_groups, \
    log_images, min_max_norm


class EMA():
    """Exponential Moving Average used for the parameters during DDPM training."""
    def __init__(self, beta:float=0.999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old:torch.Tensor, new:torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class TrainerDDPM(Trainer):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='tmp', mute:bool=True, n_channels:int=None):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute, n_channels=n_channels)
        # set train loader as a cycle instead
        self.train_loader = cycle(self.train_loader)

        # initialize step count used for training
        self.step = 0

        # specific DDPM trainer stuff
        self.gradient_accumulate_every = 2
        self.save_and_sample_every = 1000

        # EMA model for DDPM training...
        self.step_start_ema = 2000
        self.update_ema_every = 10
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model)
        self.reset_ema()

        # update name to include the depth of the model
        self.name += f'_{config["T"]}'

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

    def save_model(self, save_path:str) -> None:
        """Save the state dict of the model and ema model."""
        save_data = {
            'model': self.model.state_dict(),
            'ema_model': self.ema_model.state_dict(),
            'config': self.config,
        }
        torch.save(save_data, save_path)

    def load_model(self, save_path:str) -> None:
        """Load the state dict into the instantiated model and ema model."""
        save_data = torch.load(save_path)
        self.model.load_state_dict(save_data['model'])
        self.ema_model.load_state_dict(save_data['ema_model'])
        self.config = save_data['config']

    def sample(self) -> torch.Tensor:
        """Generate n_images samples from the model."""
        batches = num_to_groups(self.n_samples, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
        samples = torch.cat(all_images_list, dim=0)
        return samples

    def recon(self, x:torch.Tensor) -> torch.Tensor:
        """Generate n_images reconstructions from the model."""
        assert x.shape[0] >= self.n_samples
        x = x[:self.n_samples]
        x_recon = self.model.reconstruct(x)
        return x_recon

    def log_images(self, x:torch.Tensor) -> None:
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

        # log images to wandb
        step = self.step // self.save_and_sample_every
        log_images(
            x_recon=recon, 
            x_sample=samples, 
            folder=self.res_folder, 
            name=f'{step}_{self.name}_{self.config["dataset"]}.png', 
            nrow=self.n_rows
        )

    def train(self) -> torch.Tensor:
        # Instantiate wandb run
        wandb.init(project=self.wandb_name, config=self.config)
        wandb.watch(self.model)

        # run training
        losses = self.train_loop()

        # Finalize training
        self.finalize()
        return losses
    
    def train_loop(self) -> list:
        losses = []
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
            loss_obj = np.mean(train_obj)
            is_milestone = self.step != 0 and self.step % self.save_and_sample_every == 0
            wandb.log({
                'train_obj': loss_obj
            }, commit=(not is_milestone))
            if is_milestone:
                self.log_images(x)

            # update step
            self.step += 1
        
        return losses

class TrainerDownsampleDDPM(TrainerDDPM):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='', mute:bool=True, n_channels:int=None):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute, n_channels)

    def train_loop(self):
        losses = []
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
            losses.append(loss_)

            # update gradients
            self.opt.step()
            self.opt.zero_grad()

            # update EMA
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # log stuff to wandb
            is_milestone = self.step != 0 and self.step % self.save_and_sample_every == 0
            wandb.log({
                'train_obj': loss_,
                'train_latent': loss_latent_,
                'train_recon': loss_recon_
            }, commit=(not is_milestone))
            if is_milestone:
                self.log_images(x)

            # update step
            self.step += 1
        return losses