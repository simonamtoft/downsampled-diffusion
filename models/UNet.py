import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import center_crop


class UNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        in_channels = 1
        n_labels = 1
        channels = config['channels']

        # encoder (downsampling)
        enc_dims = [in_channels, *channels]
        self.enc_conv = nn.ModuleList([])
        self.enc_pool = nn.ModuleList([])
        for i in range(1, len(enc_dims)):
            module_list = append_layer([], enc_dims[i-1], enc_dims[i], config)
            for _ in range(config['n_convs']-1):
                module_list = append_layer(module_list, enc_dims[i], enc_dims[i], config)

            self.enc_conv.append(nn.Sequential(*module_list))
            self.enc_pool.append(
                 nn.Conv2d(enc_dims[i], enc_dims[i], kernel_size=2, stride=2)
            )

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_dims[i], 2*enc_dims[i], kernel_size=3, padding=config['padding']),
            nn.Conv2d(2*enc_dims[i], 2*enc_dims[i], kernel_size=3, padding=config['padding']),
            nn.Conv2d(2*enc_dims[i], enc_dims[i], kernel_size=3, padding=config['padding'])
        )

        # decoder (upsampling)
        dec_dims = [*channels]
        dec_dims.reverse()
        self.dec_conv = nn.ModuleList([])
        self.dec_upsample = nn.ModuleList([])
        for i in range(1, len(dec_dims)):
            module_list = append_layer([], 2*dec_dims[i-1], dec_dims[i], config)
            for _ in range(config['n_convs']-1):
                module_list = append_layer(module_list, dec_dims[i], dec_dims[i], config)            
            self.dec_conv.append(nn.Sequential(*module_list))
            self.dec_upsample.append(
                nn.ConvTranspose2d(dec_dims[i-1], dec_dims[i-1], kernel_size=4, stride=2, padding=1)
            )
        # final layer is without ReLU activation.
        self.dec_conv.append(nn.Sequential(
            nn.Conv2d(2*dec_dims[i], dec_dims[i], kernel_size=3, padding=config['padding']),
            nn.Conv2d(dec_dims[i], n_labels, kernel_size=3, padding=config['padding']),
        ))
        self.dec_upsample.append(
            nn.ConvTranspose2d(dec_dims[i], dec_dims[i], kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        enc = x # x is input of first encoder
        
        # Pass through the encoder
        enc_out = []
        for i in range(len(self.enc_conv)):
            # Pass through a single encoder layer
            enc = self.enc_conv[i](enc)

            # save the encoder output such that it can be used for skip connections
            enc_out.append(enc)

            # Downsample with convolutional pooling
            enc = self.enc_pool[i](enc)

        # Pass through the bottleneck
        b = self.bottleneck(enc)

        # Pass through the decoder
        dec = b
        enc_out.reverse()   # reverse such that it fits pass through decoder
        for i in range(len(self.dec_conv)):
            # Get input for decoder
            dec = self.dec_upsample[i](dec)
            enc = enc_out[i]
            dec = skip_connection(enc, dec)

            # Pass through a single decoder layer
            dec = self.dec_conv[i](dec)
        
        return dec


def skip_connection(enc, dec):
    dec = center_crop(dec, output_size=enc.shape[2:3])
    return torch.cat([enc, dec], 1)


def append_layer(module_list, dim_1, dim_2, config):
    out_list = module_list

    # Add convolutional layer
    out_list.append(
        nn.Conv2d(dim_1, dim_2, kernel_size=3, padding=config['padding'])
    )

    # add batch norm
    if config['batch_norm']:
        out_list.append(
            nn.BatchNorm2d(dim_2)
        )

    # add activation
    out_list.append(nn.ReLU())

    # add dropout
    if config['dropout']:
        out_list.append(nn.Dropout(p=config['dropout']))
    
    return out_list


class UNetSimple(nn.Module):
    def __init__(self):
        super(UNetSimple, self).__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(1, 64, 3, padding=1)
        self.pool0 = nn.Conv2d(64, 64, 2, padding=0, stride=2)
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.Conv2d(64, 64, 2, padding=0, stride=2)
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.Conv2d(64, 64, 2, padding=0, stride=2)
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.Conv2d(64, 64, 2, padding=0, stride=2)

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(64, 64, 4, stride=2)
        self.dec_conv0 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(64, 64, 4, stride=2)
        self.dec_conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(64, 64, 4, stride=2)
        self.dec_conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.upsample3 = nn.ConvTranspose2d(64, 64, 4, stride=2)
        self.dec_conv3 = nn.Conv2d(128, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        e0 = F.relu(self.enc_conv0(x))
        e0p = self.pool0(e0)
        e1 = F.relu(self.enc_conv1(e0p))
        e1p = self.pool1(e1)
        e2 = F.relu(self.enc_conv2(e1p))
        e2p = self.pool2(e2)
        e3 = F.relu(self.enc_conv3(e2p))
        e3p = self.pool3(e3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3p))

        # decoder
        d0 = F.relu(self.dec_conv0(skip_connection(e3, self.upsample0(b))))
        d1 = F.relu(self.dec_conv1(skip_connection(e2, self.upsample1(d0))))
        d2 = F.relu(self.dec_conv2(skip_connection(e1, self.upsample2(d1))))
        d3 = self.dec_conv3(skip_connection(e0, self.upsample3(d2)))  # no activation
        return d3