import torch
from torch import nn, tensor
from copy import deepcopy
from collections import OrderedDict


class EMA():
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
        
        # setup ema model
        self.ema_model = deepcopy(model)
        for param in self.ema_model.parameters():
            param.detach_()
    
    def state_dict(self):
        return self.ema_model.state_dict()
    
    def eval(self):
        self.ema_model.eval()
    
    def reset(self, model):
        self.ema_model = deepcopy(model)

    def update(self, model):
        for curr_params, new_params in zip(self.ema_model.parameters(), model.parameters()):
            curr_weight, new_weight = curr_params.data, new_params.data
            curr_params.data = self.__update_avg(curr_weight, new_weight)
    
    def __update_avg(self, curr, new):
        if curr is None:
            return new
        return curr * self.decay + (1 - self.decay) * new

    def forward(self, x:tensor) -> tensor:
        return self.ema_model(x)

    @torch.no_grad()
    def sample(self, n:int) -> tensor:
        return self.ema_model.sample(n)

    @torch.no_grad()
    def reconstruct(self, x:tensor, n:int) -> tensor:
        return self.ema_model.reconstruct(x, n)


# class EMA(nn.Module):
#     def __init__(self, model: nn.Module, decay:float=0.99):
#         """Exponential Moving Average of model parameters.

#         Args:
#             model (nn.Module):  An instantiate model, of which
#                                 we want to hold the exponential
#                                 moving average parameters.
#             decay (float):      The decaying parameter. 
#                                 A value of e.g. 0.99 means that 99% of 
#                                 the previous model weights is kept at 
#                                 each update.
#         Reference:
#         https://www.zijianhu.com/post/pytorch/ema/
#         """
#         super().__init__()
#         self.decay = decay
#         self.model = model
#         self.shadow = deepcopy(model)

#         for param in self.shadow.parameters():
#             param.detach_()

#     def update(self) -> None:
#         """Update the parameters of the shadow model"""
#         if not self.training:
#             print("EMA uopdate should only be called during training!")
#             return None

#         # update params
#         model_params = OrderedDict(self.model.named_parameters())
#         shadow_params = OrderedDict(self.shadow.named_parameters())
#         assert model_params.keys() == shadow_params.keys()
#         for name, param in model_params.items():
#             shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

#         # copy buffers
#         model_buffers = OrderedDict(self.model.named_buffers())
#         shadow_buffers = OrderedDict(self.shadow.named_buffers())
#         assert model_buffers.keys() == shadow_buffers.keys()
#         for name, buffer in model_buffers.items():
#             shadow_buffers[name].copy_(buffer)

#     def forward(self, x:tensor) -> tensor:
#         """
#         At train time forward the trained model.
#         At test time forward the shadow model.
#         """
#         if self.training:
#             return self.model(x)
#         else:
#             return self.shadow(x)
    
#     @torch.no_grad()
#     def sample(self, n:int) -> tensor:
#         if self.training:
#             return self.model.sample(n)
#         else:
#             return self.shadow.sample(n)
    
#     @torch.no_grad()
#     def reconstruct(self, x:tensor, n:int) -> tensor:
#         if self.training:
#             return self.model.reconstruct(x, n)
#         else:
#             return self.shadow.reconstruct(x, n)
