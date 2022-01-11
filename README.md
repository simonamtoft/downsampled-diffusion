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

## References

- Fisher Yu, Ari Seff, Yinda Zhang, Shuran Song, Thomas Funkhouser & Jianxiong Xiao: LSUN Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop, [arXiv:1506.03365](https://arxiv.org/abs/1506.03365)
- Alex Krizhevsky, 2009, [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), *Chapter 3*
  - [CIFAR data](https://www.cs.toronto.edu/~kriz/cifar.html)
