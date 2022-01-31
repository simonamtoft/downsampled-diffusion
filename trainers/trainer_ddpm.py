import os
import copy
import wandb
import torch
import numpy as np
from torchvision import utils

from .trainer import Trainer
from .train_helpers import cycle, num_to_groups


class EMA():
    """Exponential Moving Average used for the parameters during DDPM training."""
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class TrainerDDPM(Trainer):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='', mute:bool=True):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute)
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
        self.name += f'_{config["timesteps"]}'
        
    def reset_ema(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_ema()
            return
        self.ema.update_model_average(self.ema_model, self.model)
    
    def sample(self, n_images=36):
        """Generate n_images samples from the model."""
        batches = num_to_groups(n_images, self.batch_size)
        all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
        all_images = torch.cat(all_images_list, dim=0)
        samples = (all_images + 1) * 0.5
        return samples, int(np.sqrt(n_images))
    
    def train_loop(self):
        losses = []
        while self.step < self.n_steps:
            train_loss = []
            for _ in range(self.gradient_accumulate_every):
                # retrieve a batch and port to device
                x, _ = next(self.train_loader)
                x = x.to(self.device)
                
                # perform a model forward pass
                loss = self.model(x)

                # backward pass
                objective = loss / self.gradient_accumulate_every
                objective.backward()

                # save loss
                train_loss.append(loss.item())
            
            # update gradients
            self.opt.step()
            self.opt.zero_grad()

            # update EMA
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            # sample and log sample to wandb
            is_milestone = self.step != 0 and self.step % self.save_and_sample_every == 0
            if is_milestone:
                samples, nrows = self.sample()
                
                # save image
                step = self.step // self.save_and_sample_every
                img_path = f'{self.res_folder}/sample_{step}_{self.name}_{self.config["dataset"]}.png'
                utils.save_image(samples, img_path, nrow=nrows)
                wandb.log({'sample': wandb.Image(img_path)}, commit=False)
            
            # log loss to wandb
            loss_ = np.mean(train_loss)
            losses.append(loss_)
            wandb.log({
                'train_loss': loss_,
            }, commit=True)

            # remove local image save
            if is_milestone:
                os.remove(img_path)

            # update step
            self.step += 1
        return losses
    
    def train(self):
        # Instantiate wandb run
        wandb.init(project=self.wandb_name, config=self.config)
        wandb.watch(self.model)

        # run training
        losses = self.train_loop()
        
        # Finalize training
        torch.save(self.model, f'{self.res_folder}/{self.name}_model.pt')
        wandb.save(f'{self.res_folder}/{self.name}_model.pt')
        wandb.finish()
        print(f"Training of {self.name} completed!")
        return losses


class TrainerDownsampleDDPM(TrainerDDPM):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='', mute:bool=True):
        super().__init__(config, model, train_loader, val_loader, device, wandb_name, mute)
        
    def train_loop(self):
        losses = []
        while self.step < self.n_steps:
            train_loss = []
            train_latent = []
            train_recon = []
            for _ in range(self.gradient_accumulate_every):
                # retrieve a batch and port to device
                x, _ = next(self.train_loader)
                x = x.to(self.device)
                
                # perform a model forward pass
                loss_latent, loss_recon = self.model(x)
                loss = loss_latent + loss_recon

                # backward pass
                objective = loss / self.gradient_accumulate_every
                objective.backward()

                # save loss
                train_loss.append(loss.item())
                train_latent.append(loss_latent.item())
                train_recon.append(loss_recon.item())
            
            # update gradients
            self.opt.step()
            self.opt.zero_grad()

            # update EMA
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            # sample and log sample to wandb
            img_path, is_milestone = self.sample()
            
            # log loss to wandb
            loss_ = np.mean(train_loss)
            losses.append(loss_)
            wandb.log({
                'train_loss': loss_,
                'train_latent': np.mean(train_latent),
                'train_recon': np.mean(train_recon),
            }, commit=True)

            # remove local image save
            if is_milestone:
                os.remove(img_path)

            # update step
            self.step += 1
        return losses