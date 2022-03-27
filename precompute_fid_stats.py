# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
# Reference: 
# https://github.com/NVlabs/NVAE/blob/master/scripts/precompute_fid_statistics.py
import os
import argparse
from itertools import chain

from utils import DATASETS, get_dataloader
from fid.inception import InceptionV3
from fid.fid_score import compute_statistics_of_generator, save_statistics

FID_DIR = './results/fid_stats'
DATA_ROOT = '../data'


def main(args):
    config = {'dataset': args.dataset, 'batch_size': args.batch_size}
    device = 'cuda'
    dims = 2048
    # for binary datasets including MNIST and OMNIGLOT, we don't apply binarization for FID computation
    train_queue, valid_queue = get_dataloader(config, 'cuda', True, DATA_ROOT, val_split=0.15)
    print('len train queue', len(train_queue), 'len val queue', len(valid_queue), 'batch size', args.batch_size)
    if args.dataset in {'celeba', 'celeba_hq', 'omniglot'}:
        train_queue = chain(train_queue, valid_queue)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=FID_DIR).to(device)
    m, s = compute_statistics_of_generator(train_queue, model, args.batch_size, dims, device, args.max_samples)
    file_path = os.path.join(FID_DIR, f'{args.dataset}_{args.max_samples}.npz')
    print('saving fid stats at %s' % file_path)
    save_statistics(file_path, m, s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument(
        '--dataset', 
        type=str, 
        default='cifar10',
        choices=DATASETS,
        help='which dataset to compute FID statistics on'
    )
    # parser.add_argument(
    #     '--data', 
    #     type=str, 
    #     default='/tmp/nvae-diff/data',
    #     help='location of the data corpus'
    # )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=64, 
        help='batch size per GPU'
    )
    parser.add_argument(
        '--max_samples', 
        type=int, 
        default=50000
    )

    args = parser.parse_args()
    args.distributed = False
    main(args)