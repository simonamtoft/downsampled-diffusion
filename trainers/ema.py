import torch
import torch.nn as nn
from torch import Tensor
from copy import deepcopy
from collections import OrderedDict


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay:float=0.99):
        """Exponential Moving Average of model parameters.

        Args:
            model (nn.Module):  An instantiate model, of which
                                we want to hold the exponential
                                moving average parameters.
            decay (float):      The decaying parameter. 
                                A value of e.g. 0.99 means that 99% of 
                                the previous model weights is kept at 
                                each update.
        Reference:
        https://www.zijianhu.com/post/pytorch/ema/
        """
        super().__init__()
        self.decay = decay
        self.model = model
        self.shadow = deepcopy(model)

        for param in self.shadow.parameters():
            param.detach_()

    def update(self) -> None:
        """Update the parameters of the shadow model"""
        if not self.training:
            print("EMA uopdate should only be called during training!")
            return None

        # update params
        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())
        assert model_params.keys() == shadow_params.keys()
        for name, param in model_params.items():
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        # copy buffers
        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())
        assert model_buffers.keys() == shadow_buffers.keys()
        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def forward(self, x:Tensor) -> Tensor:
        """
        At train time forward the trained model.
        At test time forward the shadow model.
        """
        if self.training:
            return self.model(x)
        else:
            return self.shadow(x)
    
    @torch.no_grad()
    def sample(self, n:int) -> Tensor:
        if self.training:
            return self.model.sample(n)
        else:
            return self.shadow.sample(n)
    
    @torch.no_grad()
    def reconstruct(self, x:Tensor, n:int) -> Tensor:
        if self.training:
            return self.model.reconstruct(x, n)
        else:
            return self.shadow.reconstruct(x, n)