# ddpm

## Package Requirements

Start by installing some required packages listed in `requirements.txt` with

```cli
pip install -r requirements.txt
```

Then install torch for GPU (CUDA 11.1) with:
```cli
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Data Requirements

<!-- Start by downloading the LSUN datasets `tower` and `church_outdoor` and the test set, by running the following commands inside the `data` folder:

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
``` -->

The rest of the datasets is downloaded using the `Explore Datasets.ipynb` python notebook.

<!-- If you want to perform mixed precision training, a couple of things are required. Firstly, your system is required to have `nvcc` (the NVIDIA CUDA compiler), which can be installed from the [NVIDIA developer website](https://developer.nvidia.com/cuda-downloads) and following on-screen instructions. Note that the torch installation above uses [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive), which then should be downloaded instead. Secondly, you should download NVIDIAs `apex` package, which is done by:

```cli
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
``` -->

## Additional Notes

### Clean up local wandb files
If you are running weight and biases (wandb) as done in this project, it might be a good idea to once in a while delete local files that are synced to the website with the following command `wandb sync --clean`.

## References

- Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser & Jianxiong Xiao: LSUN Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop, [arXiv:1506.03365](https://arxiv.org/abs/1506.03365)
- Alex Krizhevsky, 2009, [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), *Chapter 3*
- Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015. [arXiv:1409.0575](https://arxiv.org/abs/1409.0575)
- Patryk Chrabaszcz, Ilya Loshchilov & Frank Hutter: A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets, [arXiv:1707.08819](https://arxiv.org/abs/1707.08819)
