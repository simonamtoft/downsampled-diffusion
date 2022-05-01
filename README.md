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

This GitHub repository uses the following datasets: `MNIST`, `CIFAR10`, `CelebA` and `CelebAMask-HQ`. The `MNIST` and `CIFAR10` datasets are simply downloaded using the `download=True` argument in their respective `torchvision.datasets` class. The `CelebA` dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and then resized to `64 x 64`, while the `CelebAMask-HQ` is downloaded from [a GitHub repo](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch) where two instances are created by resizing the images to `256 x 256` and `64 x 64` respectively. For easy download you should be able to download all the datasets from [my Google Drive folder](https://drive.google.com/drive/folders/15sfoeQOmZ3DyEEeV4qfIe1GfRvarLkoG?usp=sharing).

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
- Yann LeCun, Corinna Cortes, Christopher J.C. Burges, [The MNIST Database](http://yann.lecun.com/exdb/mnist/)
- Alex Krizhevsky, 2009, [Learning Multiple Layers of Features from Tiny Images, Chapter 3](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf)
- Ziwei Liu, Ping Luo, Xiaogang Wang, Xiaoou Tang, 2014, [Deep Learning Face Attributes in the Wild, arXiv:1411.7766](https://arxiv.org/abs/1411.7766v3)
- Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen, 2017, [Progressive Growing of GANs for Improved Quality, Stability, and Variation, arXiv:1710.10196](https://arxiv.org/abs/1710.10196v3)
- Cheng-Han Lee, Ziwei Liu, Lingyun Wu, Ping Luo, 2019, [MaskGAN: Towards Diverse and Interactive Facial Image Manipulation, arXiv:1907.11922](https://arxiv.org/abs/1907.11922)
- https://github.com/lucidrains/denoising-diffusion-pytorch
- https://github.com/hojonathanho/diffusion
- https://github.com/openai/guided-diffusion
- https://github.com/openai/improved-diffusion
