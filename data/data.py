from torchvision.transforms import Compose, Resize, \
    CenterCrop, ToTensor, Normalize
from torchvision.datasets import CelebA, CIFAR10, CIFAR100
from torch.utils.data import DataLoader

DATA_ROOT = './data/'


def download_datasets(data_root=DATA_ROOT) -> None:
    """Downloads the following datasets: Cifar10, Cifar100, ..."""
    # _ = CelebA(DATA_ROOT, download=True)
    print('Downloading CIFAR10')
    _ = CIFAR10(data_root, download=True)
    print('Downloading CIFAR100')
    _ = CIFAR100(data_root, download=True)


def get_dataloader(config, train=True, data_root=DATA_ROOT) -> DataLoader:
    """Returns dataloader for dataset specified in config"""

    # Define transforms to perform on each individual image
    data_transform = [
        ToTensor(),
        Resize(config['image_size']),
        CenterCrop(config['image_size']),
        # Normalize()
    ]
    data_transform = Compose(data_transform)

    # Get data
    data_args = {'download': False, 'transform': data_transform, 'train': train}
    if config['dataset'] == 'cifar10':
        data = CIFAR10(data_root, **data_args)
    elif config['dataset'] == 'cifar100':
        data = CIFAR100(data_root, **data_args)
    else:
        raise Exception(f'Dataset {config["dataset"]} not implemented...')

    # Return DataLoader for data
    return DataLoader(
        data, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        drop_last=True
    )


def get_label_map(dataset):
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
    else:
        raise Exception(f'Dataset {dataset} does not have a label map implemented...')
