# Downsampled Denoising Diffusion Probabilistic Models

## Package Requirements

Start by installing some required packages listed in `requirements.txt` with

```cli
pip install -r requirements.txt
```

Then install torch for GPU (CUDA 11.1) with:
```cli
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```


## Datasets

This GitHub repository uses the following datasets: `MNIST`, `CIFAR10`, `CelebA` and `CelebAMask-HQ`. The `MNIST` and `CIFAR10` datasets are simply downloaded using the `download=True` argument in their respective `torchvision.datasets` class. The `CelebA` dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and then resized to `64 x 64`, while the `CelebAMask-HQ` is downloaded from [a GitHub repo](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch) where two instances are created by resizing the images to `256 x 256` and `64 x 64` respectively.

### Samples from `MNIST` training set
![MNIST training examples](/images/mnist.png)

### Samples from `CIFAR10` training set
![CIFAR10 training examples](/images/cifar10.png)

### Samples from `CelebA` training set
![CelebA training examples](/images/celeba.png)

### Samples from `CelebAMask-HQ 256x256` training set
![CelebAMask-HQ-256 training examples](/images/celeba_hq.png)

### Samples from `CelebAMask-HQ 64x64` training set
![CelebAMask-HQ-64 training examples](/images/celeba_hq_64.png)


## Additional Notes

### Clean up local wandb files
If you are running weight and biases (wandb) as done in this project, it might be a good idea to once in a while delete local files that are synced to the website with the following command `wandb sync --clean`.

## References

- Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser & Jianxiong Xiao: LSUN Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop, [arXiv:1506.03365](https://arxiv.org/abs/1506.03365)
- Alex Krizhevsky, 2009, [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), *Chapter 3*
- Olga Russakovsky*, Jia Deng*, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C. Berg and Li Fei-Fei. (* = equal contribution) ImageNet Large Scale Visual Recognition Challenge. IJCV, 2015. [arXiv:1409.0575](https://arxiv.org/abs/1409.0575)
- Patryk Chrabaszcz, Ilya Loshchilov & Frank Hutter: A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets, [arXiv:1707.08819](https://arxiv.org/abs/1707.08819)
