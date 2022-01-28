import os
from torch.optim import Adam


class Trainer(object):
    def __init__(self, config:dict, model, train_loader, val_loader=None, device:str='cpu', wandb_name:str='', mute:bool=True, res_folder:str='./results'):
        """Boilerplate Trainer class, instantiating standard variables.
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
        """
        
        # Extract fields from config
        self.lr = config['lr']
        self.n_steps = config['n_steps']
        self.batch_size = config['batch_size']
        self.image_size = config['image_size']

        # Setup device to run on
        self.device = device

        # define model
        self.model = model.to(self.device)

        # setup optimizer
        self.opt = Adam(self.model.parameters(), lr=self.lr)
        
        # setup DataLoaders for training and validation set
        self.train_loader = train_loader
        if val_loader != None:
            self.val_loader = val_loader
            self.has_validation = True
        else:
            self.has_validation = False
        
        # setup results folder
        self.res_folder = res_folder
        if not os.path.isdir(self.res_folder):
            os.mkdir(self.res_folder)

        # Run weight and biases if a project name for wandb is given.
        if wandb_name != '':
            self.is_wandb = True
            self.config = config
            if mute:
                os.environ["WANDB_SILENT"] = "true"
        else:
            self.is_wandb = False

    def train(self):
        raise NotImplementedError('Implement in subclass...')