import math
import torch
from torch import nn, tensor
from einops import rearrange
from models.utils import exists


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.dim = dim

    def forward(self, x:tensor) -> tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Upsample(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x:tensor) -> tensor:
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim:int):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x:tensor) -> tensor:
        return self.conv(x)


class LayerNorm(nn.Module):
    def __init__(self, dim:int, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x:tensor) -> tensor:
        std = torch.var(x, dim=1, unbiased=False, keepdim=True).sqrt()
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (std + self.eps) * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim:int, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x:tensor) -> tensor:
        x = self.norm(x)
        return self.fn(x)


class Block(nn.Module):
    def __init__(self, dim:int, dim_out:int, groups:int=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish()
        )

    def forward(self, x:tensor) -> tensor:
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim:int, dim_out:int, *, time_emb_dim=None, groups:int=8, dropout:float=0):
        super().__init__()
        
        # Instantiate time embedding
        self.mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None
        
        # instantiate dropout between the two blocks
        self.dropout = nn.Dropout(p=dropout)

        # instantiate the convolutional blocks
        self.block1 = Block(dim, dim_out, groups)
        self.block2 = Block(dim_out, dim_out, groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x:tensor, time_emb:tensor) -> tensor:
        h = self.block1(x)

        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None]
        
        if self.dropout:
            h = self.dropout(h)
        
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x:tensor) -> tensor:
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads=self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)
