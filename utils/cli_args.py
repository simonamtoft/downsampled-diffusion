import argparse


def get_args(config: dict, data_names: list, model_names: list) -> tuple:
    parser = argparse.ArgumentParser(description="Model training script.")

    # Pick Model
    parser.add_argument(
        '-m', 
        help=f'Pick which model to train (default: {model_names[0]}).',
        default=model_names[0],
        type=str,
        choices=model_names,
        dest='model'
    )

    # Pick Dataset
    parser.add_argument(
        '-d', 
        help=f'Pick which dataset to fit to (default: {data_names[0]}).', 
        default=data_names[0],
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
    
    # Pick batch size
    parser.add_argument(
        '-bs',
        help='Pick batch size of data.', 
        default=32,
        type=int,
        dest='batch_size'
    )
    
    # Pick image size
    parser.add_argument(
        '-is',
        help='Pick image size of data.', 
        default=32,
        type=int,
        dest='image_size'
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

    # Pick whether to downsample or not (only for ddpm)
    if 'ddpm' in model_names:
        parser.add_argument(
            '-downsample',
            help='Determine how many downsamples (x2) to perform. When 0, run standard DDPM.',
            default=0,
            type=int,
            dest='n_downsamples',
        )

    # Parse the arguments
    args = parser.parse_args()

    # add args to config
    for key, value in vars(args).items():
        if key not in ['mute', 'n_runs']:           
            config[key] = value

    # remove downsample if model is not ddpm
    if config['model'] != 'ddpm':
        del config['n_downsamples']
    
    # leftover args
    args_ = {'n_runs': args.n_runs, 'mute': args.mute}

    return config, args_
