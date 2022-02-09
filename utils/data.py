import torch
import numpy as np
from torchvision.transforms import Compose, Resize, \
    CenterCrop, ToTensor, RandomHorizontalFlip, \
    Normalize, Lambda
from torchvision.datasets import CelebA, CIFAR10, \
    CIFAR100, MNIST, Omniglot
from torch.utils.data import DataLoader

DATA_ROOT = './data/'
DATASETS = ['cifar10', 'cifar100', 'mnist', 'omniglot']


def binarize(x):
    """Used as lambda function to binarize data x"""
    return torch.bernoulli(x)


def inv_binarize(x):
    """
    Used as lambda function to binarize data x 
    and reverse black and white
    """
    return 1 - torch.bernoulli(x)


def download_datasets(data_root:str=DATA_ROOT) -> None:
    """Downloads the following datasets: CIFAR10, CIFAR100, Omniglot & MNIST"""
    # _ = CelebA(DATA_ROOT, download=True)
    print('Downloading CIFAR10')
    _ = CIFAR10(data_root, download=True)
    print('Downloading CIFAR100')
    _ = CIFAR100(data_root, download=True)
    print('Downloadning MNIST')
    _ = MNIST(data_root, download=True)
    print('Downloadning Omniglot')
    _ = Omniglot(data_root, download=True)
    print('Finished downloading CIFAR10, CIFAR100, Omniglot & MNIST...')


def get_transforms(config:dict):
    """Define transforms to use, based on model and dataset."""
    
    # get dataset and model name from config
    dataset = config['dataset']
    model = config['model']
    size = config['image_size']
    
    # setup standard transform
    data_transform = [
        Resize(size),
        CenterCrop(size),
        ToTensor(),
    ]
    
    # add binarization for autoencoder models on mnist and omniglot.
    if model in ['vae', 'draw']:
        if dataset == 'mnist':
            data_transform.append(Lambda(binarize))
        elif dataset == 'omniglot':
            data_transform.append(Lambda(inv_binarize))
    # add flip for ddpm model on datasets such as cifar, celeba etc.
    elif model == 'ddpm' and dataset not in ['mnist', 'omniglot']:
        data_transform.append(RandomHorizontalFlip())
    
    return data_transform   


def get_dataloader(config:dict, device:str, train:bool=True, data_root:str=DATA_ROOT, val_split:float=0.15) -> DataLoader:
    """ 
    Returns dataloaders for train and validation splits of the dataset specified in config.
        
    Args
        config (dict):      A dict determining the 'dataset' to use and the resulting 
                            'image_size' of the images in the resulting DataLoader.
        train (bool):       A boolean value that determines whether to take the train or test set.
        data_root (str):    A string that describes the path to the downloaded data.
        val_split (float):  A float that determines how much of the data that should be in 
                            the validation set [0; 1]. Ignored if train = False.
    
    Returns
        train_set (DataLoader): If train=True, this dataloader contains the shuffled training
                                samples.
        val_set (DataLoader):   If train=True, this DataLoader contains the validation samples.
        test_set (DataLoader):  If train=False, this DataLoader contains the test samples.
    """

    # Define transforms to perform on each individual image
    data_transform = Compose(get_transforms(config))
    
    # Initialize data arguments
    data_args = {'download': False, 'transform': data_transform, 'train': train}

    # Get data
    if config['dataset'] == 'cifar10':
        data = CIFAR10(data_root, **data_args)
    elif config['dataset'] == 'cifar100':
        data = CIFAR100(data_root, **data_args)
    elif config['dataset'] == 'mnist':
        data = MNIST(data_root, **data_args)
    elif config['dataset'] == 'omniglot':
        data = Omniglot(data_root, **data_args)
    else:
        raise Exception(f'Dataset {config["dataset"]} not implemented...')

    # setup CUDA args for DataLoaders
    kwargs = {'num_workers': 4, 'pin_memory': True} if device == 'cuda' else {}
    
    # return train and validation DataLoaders
    if train:
        # define number of samples in train and validation sets
        split = (len(data) * np.array([1-val_split, val_split])).astype(int)

        # split data into train and validation
        train_data, val_data = torch.utils.data.random_split(data, list(split))

        # Create and return dataloaders for each set
        train_set = DataLoader(
            train_data,
            batch_size=config['batch_size'], 
            shuffle=True, 
            drop_last=True,
            **kwargs,
        )
        val_set = DataLoader(
            val_data,
            batch_size=config['batch_size'], 
            shuffle=False, 
            drop_last=True,
            **kwargs,
        )
        return train_set, val_set
    
    # return test DataLoader
    else:
        test_set = DataLoader(
            data,
            batch_size=config['batch_size'], 
            shuffle=False, 
            drop_last=True,
            **kwargs,
        )
        return test_set


def get_color_channels(dataset:str) -> int:
    if dataset in ['cifar10', 'cifar100']:
        return 3
    elif dataset in ['mnist', 'omniglot']:
        return 1
    else:
        raise Exception(f'Dataset {dataset} does not have a color channel set...')


def get_label_map(dataset:str) -> list:
    if dataset == 'cifar10':
        return [
            'airplane', 'automobile', 'bird', 
            'cat', 'deer', 'dog', 'frog', 
            'horse', 'ship', 'truck'
        ]
    elif dataset == 'cifar100':
        return [
            'apple', 'aquarium fish', 'baby', 'bear', 'beaver', 
            'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 
            'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 
            'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 
            'clock', 'cloud', 'cockroach', 'couch', 'cra', 'crocodile', 
            'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 
            'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 
            'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion', 'lizard',
            'lobster', 'man', 'maple tree', 'motorcycle', 'mountain', 
            'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 
            'palm tree', 'pear', 'pickup truck', 'pine tree', 'plain', 
            'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
            'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 
            'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 
            'squirrel', 'streetcar', 'sunflower', 'sweet pepper', 'table',
            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 
            'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 
            'wolf', 'woman', 'worm'
        ]
    elif dataset == 'mnist':
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    else:
        raise Exception(f'Dataset {dataset} does not have a label map implemented...')
