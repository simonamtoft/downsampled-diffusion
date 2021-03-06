import os
import json
import torch

from trainers import setup_trainer
from utils import CHECKPOINT_DIR, DATA_DIR

WANDB_PROJECT = 'ddpm-test'
CHECKPOINT_NAME = 'celeba_x2_3.pt'

if __name__ == '__main__':
    # load checkpoint
    print(f'Loading checkpoint {CHECKPOINT_NAME}')
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, CHECKPOINT_NAME))
    config = checkpoint['config']
    trainer, config = setup_trainer(config, True, DATA_DIR, WANDB_PROJECT, seed=0)
    trainer.load_checkpoint(checkpoint)

    # start training from checkpoint
    print(f'Starting at step {checkpoint["step"]}.')
    print('Using configuration dict:')
    print(json.dumps(config, sort_keys=False, indent=4) + '\n')
    trainer.train()
    print("train_from_checkpoint.py script finished!")
