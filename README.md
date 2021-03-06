# Downsampled Denoising Diffusion Probabilistic Models
Denoising diffusion probabilistic models (DDPM) are deep generative models that generate high quality images achieving state-of-the-art Fréchet inception distance (FID) and inception scores (IS) on benchmark datasets. However, sampling from the models is extremely time consuming compared to VAE, GAN and Flow-based models, making it computationally challenging to gain the full potential of these models on large datasets. This thesis presents an investigation of the balance between sampling time and image quality by analysing the effect of input dimensionality and model size. Based on these insights, this work introduces downsampled denoising diffusion probabilistic models (dDDPM), which adds light-weight downsampling and upsampling networks around the standard unconditional DDPM implementation. The dDDPMs achieve between 29 and 266 times faster sampling on $256\times 256$ datasets compared to standard DDPM, while providing FID scores of 37.8 on CIFAR-10, 7.9 on CelebA and 20.7 on CelebAMask-HQ. These results show that it is possible to build memory and computational efficient diffusion models while preserving most of the visual quality compared to full-sized DDPMs, providing a promising direction for deep generative models in constrained environments.

![dDDPM example](/images/x4upsample.png)


## Package Requirements

Start by installing some required packages listed in `requirements.txt` with

```cli
pip install -r requirements.txt
```

Then install torch for GPU (CUDA 11.1) with:
```cli
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Scripts
Training of a dDDPM is performed by running `train.py`, where hyper paremeters can be altered in the config dict. To run a dDDPM x3 for 800k training steps on CelebAMask-HQ with a batch size of 32 use the following
```cli
python train.py -m ddpm -e 800000 -mute -d celeba_hq -bs 32 -is 256 -downsample 3
```
In order to run the standard DDPM set the `-downsample` flag to 0.

Additionally, samples can be generated from a saved checkpoint of the dDDPM by running `generate_model_samples.py`, where the `saved_model` variable inside the script is defined to be the name of the checkpoint located at the `CHECKPOINT_DIR` defined in `utils.paths`.
```cli
python generate_model_samples.py
```

## Datasets

This GitHub repository uses the following datasets: `MNIST`, `CIFAR-10`, `CelebA` and `CelebAMask-HQ`. The `MNIST` and `CIFAR-10` datasets are simply downloaded using the `download=True` argument in their respective `torchvision.datasets` class. The `CelebA` dataset is downloaded from [Kaggle](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset) and then resized to `64 x 64`, while the `CelebAMask-HQ` is downloaded from [a GitHub repo](https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch) which is rescaled `256 x 256`. For easy download you should be able to download all the datasets from [my Google Drive folder](https://drive.google.com/drive/folders/15sfoeQOmZ3DyEEeV4qfIe1GfRvarLkoG?usp=sharing).

### Samples from `MNIST` training set
![MNIST training examples](/images/mnist.png)

### Samples from `CIFAR-10` training set
![CIFAR-10 training examples](/images/cifar10.png)

### Samples from `CelebA` training set
![CelebA training examples](/images/celeba.png)

### Samples from `CelebAMask-HQ 256x256` training set
![CelebAMask-HQ-256 training examples](/images/celeba_hq.png)

## Acknowledgements
I would like to thank [Ole Winther](https://olewinther.github.io/) and [Giorgio Giannone](https://georgosgeorgos.github.io/) for supervising this project and for all their insightful comments and suggestions throughout the course of this thesis.

## Additional Notes
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
