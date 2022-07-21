import torch.nn as nn
import models.archs.arch_util as arch_util
import torch

from models.archs.arch_util import USConv2d, USConvTranspose2d


class US_FSRCNN_net(torch.nn.Module):
    def __init__(self, input_channels, upscale, nf=64, s=12, m=4):
        super(US_FSRCNN_net, self).__init__()

        self.head_conv = nn.Sequential(
            USConv2d(in_channels=input_channels, out_channels=nf, kernel_size=5, stride=1, padding=2, us=[False, True]),
            nn.PReLU()
        )

        self.layers = []
        self.layers.append(nn.Sequential(
            USConv2d(in_channels=nf, out_channels=s, kernel_size=1, stride=1, padding=0, us=[True, False]),
            nn.PReLU())
        )
        for _ in range(m):
            self.layers.append(
                USConv2d(in_channels=s, out_channels=s, kernel_size=3, stride=1, padding=1, us=[False, False])
            )
        self.layers.append(nn.PReLU())
        self.layers.append(nn.Sequential(
            USConv2d(in_channels=s, out_channels=nf, kernel_size=1, stride=1, padding=0, us=[False, True]),
            nn.PReLU())
        )

        self.body_conv = torch.nn.Sequential(*self.layers)

        # Deconvolution
        self.tail_conv = USConvTranspose2d(in_channels=nf, out_channels=input_channels, kernel_size=9, stride=upscale, padding=3, output_padding=1, us=[True, False])

        arch_util.initialize_weights([self.head_conv, self.body_conv, self.tail_conv], 0.1)

    def forward(self, x):
        fea = self.head_conv(x)
        fea = self.body_conv(fea)
        out = self.tail_conv(fea)
        return out