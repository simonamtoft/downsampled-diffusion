import os
import copy
import wandb
import torch
import numpy as np
from torchvision import utils

from .trainer import Trainer
from .train_helpers import cycle


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
        super().__init__(model, train_loader, config, device, wandb_name, mute)

        # set train loader as a cycle instead
        self.train_loader = cycle(self.train_loader)

        # initialize step count used for training
        self.step = 0

        # specific DDPM trainer stuff
        self.gradient_accumulate_every = 2
        
        # EMA model for DDPM training...
        self.step_start_ema = 2000
        self.update_ema_every = 10
        self.ema = EMA(0.995)
        self.ema_model = copy.deepcopy(self.model)
        
    def reset_ema(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
    
    def train(self):
        # Instantiate wandb run
        if self.is_wandb:
            wandb.init(project=self.wandb_name, config=self.config)
            wandb.watch(self.model)
            
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
                train_loss.append(loss)
            
            # update gradients
            self.opt.step()
            self.opt.zero_grad()

            # update EMA
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            
            # sample
            is_milestone = self.step != 0 and self.step % self.save_and_sample_every == 0
            if is_milestone:
                # compute milestone number
                milestone = self.step // self.save_and_sample_every

                # generate samples
                batches = num_to_groups(36, self.batch_size)
                all_images_list = list(map(lambda n: self.ema_model.sample(batch_size=n), batches))
                all_images = torch.cat(all_images_list, dim=0)
                all_images = (all_images + 1) * 0.5

                # log images to wandb
                img_path = str(self.results_folder / f'sample-{milestone}-{self.config["model"]}-{self.config["dataset"]}.png')
                utils.save_image(all_images, img_path, nrow=6)
                wandb.log({'sample': wandb.Image(img_path)}, commit=False)
            
            # log loss to wandb
            wandb.log({'train_loss': np.mean(train_loss)}, commit=True)

            # remove local image save
            if is_milestone:
                os.remove(img_path)

            # update step
            self.step += 1
        
        # finish training
        wandb.finish()
        print("Training of DDPM completed!")