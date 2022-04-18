import os
import json
import torch

from trainers import setup_trainer
from utils import CHECKPOINT_DIR, DATA_DIR

WANDB_PROJECT = 'ddpm-test'
CHECKPOINT_NAME = 'chq_x3_AE_rnd_2.pt'


if __name__ == '__main__':
    # load checkpoint
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME))
    config = checkpoint['config']
    if config['model'] == 'dddpm':
        if 'u_mode' not in config:
            config['u_mode'] = config['d_mode']
    trainer, config = setup_trainer(config, True, DATA_DIR, WANDB_PROJECT, seed=0)
    trainer.load_checkpoint(checkpoint)
    
    # print out train configuration
    print(f'\nTraining from checkpoint {CHECKPOINT_NAME}')
    print(f'Starting at step {checkpoint["step"]}.')
    print('Using configuration dict:')
    print(json.dumps(config, sort_keys=False, indent=4) + '\n')
    
    # start training from checkpoint
    trainer.train()
    print("train_from_checkpoint.py script finished!")
