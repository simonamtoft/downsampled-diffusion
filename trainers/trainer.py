import os
import json
import wandb
import torch
import numpy as np
from torch.optim import Adam
from utils import reduce_mean, LOGGING_DIR


class Trainer(object):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='tmp', mute:bool=True, n_channels:int=None, n_samples:int=25):
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
        self.n_channels = n_channels        # number of color channels in data
        self.n_samples = n_samples          # number of samples to take when logging
        
        # Check that n_samples is "legal"
        self.n_rows = np.sqrt(self.n_samples).astype(int)
        if self.n_rows**2 != self.n_samples:
            raise ValueError(f'Number of samples ({self.n_samples}) has to be a square number.')
        if self.n_samples > self.batch_size:
            raise ValueError(f'Number of samples ({self.n_samples}) has to be lower than batch size ({self.batch_size}).')
        
        # mute outputs from weight and biases (wandb)
        if mute:
            os.environ["WANDB_SILENT"] = "true"
        
        # define how to log losses
        # currently not used for DDPM.
        self.loss_handle = reduce_mean
        self.train_losses = []
        
        # dimensionality of data
        self.x_dim = int(self.n_channels * self.image_size * self.image_size)

        # define model
        self.model = model.to(self.device)

        # setup optimizer
        self.opt = Adam(self.model.parameters(), lr=self.lr)      

    def save_losses(self) -> None:
        """Save loss results to file"""
        file_path = os.path.join(LOGGING_DIR, f'loss_{self.name}_{self.config["dataset"]}.json')
        print(f'Saving losses to file {file_path}')
        with open(file_path, 'w') as f:
            json.dump(self.train_losses, f)
    
    def init_wandb(self) -> None:
        # check if we are resuming run
        # https://docs.wandb.ai/guides/track/advanced/resuming
        if 'wandb_id' in self.config:
            self.wandb_id = self.config['wandb_id']
        else:
            self.wandb_id = wandb.util.generate_id()
            self.config['wandb_id'] = self.wandb_id
        
        # define checkpoint name
        self.checkpoint_name = os.path.join(LOGGING_DIR, f'checkpoint_{self.name}_{self.wandb_id}.pt')

        # Instantiate wandb run
        wandb.init(project=self.wandb_name, config=self.config, resume='allow', id=self.wandb_id)
        wandb.watch(self.model)
    
    def finalize(self) -> None:
        """Finalize training by saving the model to wandb and finishing the wandb run."""
        self.save_checkpoint()
        wandb.finish()
        os.remove(self.checkpoint_name)
        print(f"Training of {self.name} completed!")
    
    def train(self) -> torch.Tensor:
        """Run training. Starts from checkpoint if 'wandb_id' is in config."""
        self.init_wandb()
        losses = self.train_loop()
        self.finalize()
        return losses
    
    def train_loop(self) -> torch.Tensor:
        raise NotImplementedError('Implement in subclass.')

    def load_checkpoint(self):
        raise NotImplementedError('Implement in subclass.')
    
    def save_checkpoint(self):
        raise NotImplementedError('Implement in subclass.')
