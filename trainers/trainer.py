import os
import json
import wandb
import torch
import numpy as np
from torch.optim import Adam
from functools import partial

from .train_helpers import compute_bits_dim, \
    mean_and_bits_dim, nats_mean


class Trainer(object):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='tmp', mute:bool=True, res_folder:str='./results', n_channels:int=None, n_samples:int=25):
        """
        Trainer class, instantiating standard variables.
            config:         A dict that contains the necessary fields for training the specific model,
                            such as the learning rate and number of training steps/epochs to perform. 
                            This config is logged to wandb if wandb_name is set.
            model:          An instantiated torch model, (subclass of torch.nn.Module), that should be trained.
            train_loader:   A torchvision.DataLoader that contains all the train data.
            val_loader:     A torchvision.DataLoader that contains all the validation data.
            device:         A string, determining which device to perform the training on.
            wandb_name:     A string of the projectname to log to on weight and biases. 
                            If the string is empty, the run wont be logged.
            mute:           Whether to mute all prints such as tqdm during training.
            res_folder:     A string to the path of the results folder for any images etc.
            n_channels:     Number of channels in the image data. 
                            If None, it is set to 1 for mnist and omniglot, and 3 otherwise.
            n_samples:      Number of samples to generate and log as an image for each log made to wandb.
        """
        
        # Extract fields from config
        self.lr = config['lr']
        self.n_steps = config['n_steps']
        self.batch_size = config['batch_size']
        self.image_size = config['image_size']
        self.name = config['model']
        
        # store input args
        self.config = config                # logged to wandb
        self.train_loader = train_loader    # DataLoader for train data
        self.val_loader = val_loader        # DataLoader for validation data
        self.device = device                # device to run on (cpu or cuda)
        self.wandb_name = wandb_name        # name of wandb project
        self.mute = mute                    # mute prints etc.
        self.res_folder = res_folder        # store folder
        self.n_channels = n_channels        # number of color channels in data
        self.n_samples = n_samples          # number of samples to take when logging
        
        # Check that n_samples is "legal"
        self.n_rows = np.sqrt(self.n_samples).astype(int)
        if self.n_rows**2 != self.n_samples:
            raise ValueError(f'Number of samples ({self.n_samples}) has to be a square number.')
        if self.n_samples > self.batch_size:
            raise ValueError(f'Number of samples ({self.n_samples}) has to be lower than batch size ({self.batch_size}).')
        
        # create the results folder if it doesn't exist
        if not os.path.isdir(self.res_folder):
            os.mkdir(self.res_folder)
        
        # mute outputs from weight and biases (wandb)
        if mute:
            os.environ["WANDB_SILENT"] = "true"
        
        # define how to log losses
        # if self.n_channels == 1:
        self.loss_handle = nats_mean
        # else:
        #     self.loss_handle = partial(mean_and_bits_dim, self.x_dim)
    
        # dimensionality of data
        self.x_dim = int(self.n_channels * self.image_size * self.image_size)

        # define model
        self.model = model.to(self.device)

        # setup optimizer
        self.opt = Adam(self.model.parameters(), lr=self.lr)      

    def save_losses(self, losses):
        """Save loss results to file"""
        filename = f'{self.res_folder}/loss_{self.name}_{self.config["dataset"]}.json'
        print(f'Saving losses to file {filename}')
        with open(filename, 'w') as f:
            json.dump(losses, f)
    
    def save_model(self, save_path):
        """
        Save the state dict of the model.
        https://pytorch.org/tutorials/beginner/saving_loading_models.html
        """
        save_data = {
            'model': self.model.state_dict()
        }
        torch.save(save_data, save_path)

    def load_model(self, save_path):
        """Load the state dict into the instantiated model."""
        save_data = torch.load(save_path)
        self.model.load_state_dict(save_data['model'])

    def finalize(self):
        """Finalize training by saving the model to wandb and finishing the wandb run."""
        save_path = f'{self.res_folder}/model_{self.name}.pt'
        self.save_model(save_path)
        wandb.save(save_path)
        wandb.finish()
        os.remove(save_path)
        print(f"Training of {self.name} completed!")

    def get_model(self):
        return self.model
    
    def train(self):
        raise NotImplementedError('Implement in subclass...')
