import argparse


def get_args(data_names: list, config: dict) -> tuple:
    parser = argparse.ArgumentParser(description="Model training script.")

    # Pick Model
    parser.add_argument(
        '-m', 
        help='Pick which model to train (default: ddpm).',
        default='ddpm',
        type=str,
        choices=['ddpm'],
        dest='model'
    )

    # Pick Dataset
    parser.add_argument(
        '-d', 
        help='Pick which dataset to fit to (default: mnist).', 
        default='mnist',
        type=str,
        choices=data_names,
        dest='dataset'
    )

    # Pick Epochs
    parser.add_argument(
        '-e', 
        help='Pick number of epochs/trainsteps to train over (default: 500).', 
        default=500,
        type=int,
        dest='n_steps'
    )

    # Pick whether to mute all outputs or not.
    parser.add_argument(
        '-mute', 
        help='Mute tqdm and other print outputs.', 
        action='store_true'
    )

    # Pick number of models to train
    parser.add_argument(
        '-n',
        help='Pick number times to train model.', 
        default=1,
        type=int,
        dest='n_runs'
    )

    # Pick whether to downsample or not
    parser.add_argument(
        '-downsample',
        help='Train DDPM with downsampling.',
        action='store_true'
    )

    # Parse the arguments
    args = parser.parse_args()

    # add args to config
    for key, value in vars(args).items():
        if key not in ['mute', 'n_runs']:
            config[key] = value

    # leftover args
    args = {'n_runs': args.n_runs, 'mute': args.mute}

    return config, args
