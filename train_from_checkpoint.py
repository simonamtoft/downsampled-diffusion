import os
import json
import torch
from trainers import setup_trainer

DATA_ROOT = '../data/'
RES_FOLDER = './results'
WANDB_PROJECT = 'ddpm-test'
# CHECKPOINT_NAME = 'checkpoint_ddpm_vlb_paper_2.pt'
CHECKPOINT_NAME = 'checkpoint_ddpm_simple_paper_3.pt'


if __name__ == '__main__':    
    checkpoint = torch.load(os.path.join(RES_FOLDER, CHECKPOINT_NAME))
    config = checkpoint['config']
    config['loss_flat'] = 'mean'
    
    # instantiate trainer
    trainer, config = setup_trainer(config, True, DATA_ROOT, WANDB_PROJECT, RES_FOLDER, seed=0)

    # load checkpoint into trainer
    trainer.load_checkpoint(checkpoint)
    
    # print out train configuration
    print(f'\nTraining from checkpoint {CHECKPOINT_NAME} with configuration dict:')
    print(json.dumps(config, sort_keys=False, indent=4) + '\n')
    
    # start training
    trainer.train()
    
    print("train_from_checkpoint.py script finished!")
