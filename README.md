# ddpm

## Package Requirements

Start by installing some required packages listed in `requirements.txt` with

```cli
pip install -r requirements.txt
```

Then install torch for GPU with

```cli
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```

## Data Requirements

Start by downloading the LSUN datasets `tower` and `church_outdoor` and the test set, by running the following commands inside the `data` folder:

```cli
python download_lsun.py -c tower
python download_lsun.py -c church_outdoor
python download_lsun.py -c test
```

Then download the resized ImageNet 32x32 dataset, and convert it such that it can be used with torchvision, following the instructions from [NVAE GitHub](https://github.com/NVlabs/NVAE):

```cli
cd data
mkdir imagenet-oord
cd imagenet-oord
wget https://storage.googleapis.com/glow-demo/data/imagenet-oord-tfr.tar
tar -xvf imagenet-oord-tfr.tar
python convert_tfrecord_to_lmdb.py --dataset=imagenet-oord_32 --tfr_path=$DATA_DIR/imagenet-oord/mnt/host/imagenet-oord-tfr --lmdb_path=/imagenet-oord/imagenet-oord-lmdb_32 --split=train
python convert_tfrecord_to_lmdb.py --dataset=imagenet-oord_32 --tfr_path=$DATA_DIR/imagenet-oord/mnt/host/imagenet-oord-tfr --lmdb_path=/imagenet-oord/imagenet-oord-lmdb_32 --split=validation
```

The rest of the datasets is downloaded using the `Explore Datasets.ipynb` python notebook.

## References

- Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser & Jianxiong Xiao: LSUN Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop, [arXiv:1506.03365](https://arxiv.org/abs/1506.03365)
- Alex Krizhevsky, 2009, [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), *Chapter 3*
- Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015. [arXiv:1409.0575](https://arxiv.org/abs/1409.0575)
- Patryk Chrabaszcz, Ilya Loshchilov & Frank Hutter: A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets, [arXiv:1707.08819](https://arxiv.org/abs/1707.08819)