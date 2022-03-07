import torch
from torch import dropout, nn
from models.utils import exists, default
from models.unet.blocks import SinusoidalPosEmb, \
    Mish, ResnetBlock, Residual, LinearAttention, \
    PreNorm, Downsample, Upsample, Block


class Unet(nn.Module):
    def __init__(self, config:dict):
        """
        The U-Net model with attention and timestep embedding.
        Originally ported from:
        https://github.com/lucidrains/denoising-diffusion-pytorch/
        """
        # dim:int, out_dim:int=None, dim_mults:tuple=(1, 2, 4, 8), in_channels:int=3, 
        super().__init__()

        dim = config['unet_chan']
        in_channels = config['unet_in']
        dim_mults = config['unet_dims']
        dropout = config['unet_dropout']

        # create list of channels for every layer in the U-Net
        dims = [in_channels, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # enable time embedding
        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )

        # instantiate contracting and expansive paths
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # add layers to the contracting path
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim, dropout=dropout),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim, dropout=dropout),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity()
            ]))

        # add bottom layer between the contracting and expansive paths.
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # add layers to the expansive path
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity()
            ]))
        
        # final 1x1 convolutional layer
        self.final_conv = nn.Sequential(
            Block(dim, dim),
            nn.Conv2d(dim, in_channels, 1)
        )

    def forward(self, x, time):
        # Instantiate time embedding 
        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # keep track of outputs for each layer in the contracting path
        # used to concatenate in the expansive path
        h = []

        # pass through contracting path
        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        # pass through bottom layer
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # pass through expansive path
        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)
        
        # return result of final 1x1 convolution
        return self.final_conv(x)
