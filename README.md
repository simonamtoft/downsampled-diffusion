# ddpm

## Package Requirements

Start by installing some required packages listed in `requirements.txt` with

```cli
pip install -r requirements.txt
```

Then install torch for GPU (CUDA 11.0) with:

```cli
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```


<!-- If you want to perform mixed precision training, a couple of things are required. Firstly, your system is required to have `nvcc` (the NVIDIA CUDA compiler), which can be installed from the [NVIDIA developer website](https://developer.nvidia.com/cuda-downloads) and following on-screen instructions. Note that the torch installation above uses [CUDA 11.0](https://developer.nvidia.com/cuda-11.0-download-archive), which then should be downloaded instead. Secondly, you should download NVIDIAs `apex` package, which is done by:

```cli
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
``` -->

## References

- Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser & Jianxiong Xiao: LSUN Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop, [arXiv:1506.03365](https://arxiv.org/abs/1506.03365)
- Alex Krizhevsky, 2009, [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), *Chapter 3*
  - [CIFAR data](https://www.cs.toronto.edu/~kriz/cifar.html)
