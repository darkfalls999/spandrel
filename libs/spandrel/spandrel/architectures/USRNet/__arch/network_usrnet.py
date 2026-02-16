from typing import List, Tuple, Optional
import numpy as np
import torch
import torch.fft
import torch.nn as nn
from torch import Tensor

from ...__arch_helpers import basicblock as B

# for pytorch version >= 1.8.1


"""
# --------------------------------------------
# Kai Zhang (cskaizhang@gmail.com)
@inproceedings{zhang2020deep,
  title={Deep unfolding network for image super-resolution},
  author={Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={0--0},
  year={2020}
}
# --------------------------------------------
"""


"""
# --------------------------------------------
# basic functions
# --------------------------------------------
"""


def splits(a: Tensor, sf: int) -> Tensor:
    """split a into sfxsf distinct blocks

    Args:
        a: NxCxWxH
        sf: split factor

    Returns:
        b: NxCx(W/sf)x(H/sf)x(sf^2)
    """
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=4)
    b = torch.cat(torch.chunk(b, sf, dim=3), dim=4)
    return b

def p2o(psf: Tensor, shape: Tuple[int, int]) -> Tensor:
    """
    Convert point-spread function to optical transfer function.
    
    Args:
        psf: NxCxhxw
        shape: [H, W]

    Returns:
        otf: NxCxHxWx2
    """
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf

def upsample(x: Tensor, sf: int = 3) -> Tensor:
    """s-fold upsampler
    
    Args:
        x: tensor image, NxCxWxH
        sf: scale factor
    """
    st = 0
    z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*sf, x.shape[3]*sf)).type_as(x)
    z[..., st::sf, st::sf].copy_(x)
    return z

def downsample(x: Tensor, sf: int = 3) -> Tensor:
    """s-fold downsampler
    
    Args:
        x: tensor image, NxCxWxH
        sf: scale factor
    """
    st = 0
    return x[..., st::sf, st::sf]


class ResUNet(nn.Module):
    def __init__(
        self,
        in_nc: int = 4,
        out_nc: int = 3,
        nc: List[int] = [64, 128, 256, 512],
        act_mode: str = "R",
    ) -> None:
        super().__init__()
        
        self.m_head = B.conv(in_nc, nc[0], bias=False, mode="C")

        # Downsample blocks with ModuleList
        self.m_down1 = nn.ModuleList([
            B.ResBlock(nc[0], nc[0], bias=False, mode="C"+act_mode+"C"),
            B.ResBlock(nc[0], nc[0], bias=False, mode="C"+act_mode+"C"),
            nn.Conv2d(nc[0], nc[1], kernel_size=2, stride=2, bias=False)
        ])

        self.m_down2 = nn.ModuleList([
            B.ResBlock(nc[1], nc[1], bias=False, mode="C"+act_mode+"C"),
            B.ResBlock(nc[1], nc[1], bias=False, mode="C"+act_mode+"C"),
            nn.Conv2d(nc[1], nc[2], kernel_size=2, stride=2, bias=False)
        ])

        self.m_down3 = nn.ModuleList([
            B.ResBlock(nc[2], nc[2], bias=False, mode="C"+act_mode+"C"),
            B.ResBlock(nc[2], nc[2], bias=False, mode="C"+act_mode+"C"),
            nn.Conv2d(nc[2], nc[3], kernel_size=2, stride=2, bias=False)
        ])

        self.m_body = nn.ModuleList([
            B.ResBlock(nc[3], nc[3], bias=False, mode="C"+act_mode+"C"),
            B.ResBlock(nc[3], nc[3], bias=False, mode="C"+act_mode+"C")
        ])

        # Upsample blocks with ModuleList
        self.m_up3 = nn.ModuleList([
            nn.ConvTranspose2d(nc[3], nc[2], kernel_size=2, stride=2, bias=False),
            B.ResBlock(nc[2], nc[2], bias=False, mode="C"+act_mode+"C"),
            B.ResBlock(nc[2], nc[2], bias=False, mode="C"+act_mode+"C")
        ])

        self.m_up2 = nn.ModuleList([
            nn.ConvTranspose2d(nc[2], nc[1], kernel_size=2, stride=2, bias=False),
            B.ResBlock(nc[1], nc[1], bias=False, mode="C"+act_mode+"C"),
            B.ResBlock(nc[1], nc[1], bias=False, mode="C"+act_mode+"C")
        ])

        self.m_up1 = nn.ModuleList([
            nn.ConvTranspose2d(nc[1], nc[0], kernel_size=2, stride=2, bias=False),
            B.ResBlock(nc[0], nc[0], bias=False, mode="C"+act_mode+"C"),
            B.ResBlock(nc[0], nc[0], bias=False, mode="C"+act_mode+"C")
        ])

        self.m_tail = B.conv(nc[0], out_nc, bias=False, mode="C")

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.size()[-2:]
        paddingBottom = int(np.ceil(h/8)*8-h)
        paddingRight = int(np.ceil(w/8)*8-w)
        x = nn.ReplicationPad2d((0, paddingRight, 0, paddingBottom))(x)

        x1 = self.m_head(x)
        
        # Down path
        x = x1
        for layer in self.m_down1:
            x = layer(x)
        x2 = x

        x = x2
        for layer in self.m_down2:
            x = layer(x)
        x3 = x

        x = x3
        for layer in self.m_down3:
            x = layer(x)
        x4 = x

        # Body
        x = x4
        for layer in self.m_body:
            x = layer(x)

        # Up path with skip connections
        x = x + x4
        for layer in self.m_up3:
            x = layer(x)

        x = x + x3
        for layer in self.m_up2:
            x = layer(x)

        x = x + x2
        for layer in self.m_up1:
            x = layer(x)

        x = self.m_tail(x + x1)
        x = x[..., :h, :w]
        
        return x

class DataNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, FB: Tensor, FBC: Tensor, F2B: Tensor, 
                FBFy: Tensor, alpha: Tensor, sf: int) -> Tensor:
        FR = FBFy + torch.fft.fftn(alpha * x, dim=(-2, -1))
        x1 = FB.mul(FR)
        FBR = torch.mean(splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + alpha)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, sf, sf)
        FX = (FR - FCBinvWBR) / alpha
        Xest = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))
        return Xest

class HyPaNet(nn.Module):
    def __init__(self, in_nc: int = 2, out_nc: int = 8, channel: int = 64) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(in_nc, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
            nn.Softplus()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.mlp(x) + 1e-6
        return x


class USRNet(nn.Module):
    def __init__(
        self,
        n_iter: int = 8,
        h_nc: int = 64,
        in_nc: int = 4,
        out_nc: int = 3,
        nc: List[int] = [64, 128, 256, 512],
        act_mode: str = "R",
    ) -> None:
        super().__init__()
        
        self.d = DataNet()
        self.p = ResUNet(
            in_nc=in_nc,
            out_nc=out_nc,
            nc=nc,
            act_mode=act_mode
        )
        self.h = HyPaNet(in_nc=2, out_nc=n_iter*2, channel=h_nc)
        self.n = n_iter


    def forward(self, x: Tensor, k: Tensor, sf: int, sigma: Tensor) -> Tensor:
        """
        Args:
            x: tensor, NxCxWxH
            k: tensor, Nx(1,3)xwxh
            sf: integer, scale factor
            sigma: tensor, Nx1x1x1
        """
        w, h = x.shape[-2:]
        FB = p2o(k, (w*sf, h*sf))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        STy = upsample(x, sf=sf)
        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
        x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')

        ab = self.h(torch.cat((sigma, torch.tensor(sf).type_as(sigma).expand_as(sigma)), dim=1))

        for i in range(self.n):
            x = self.d(x, FB, FBC, F2B, FBFy, ab[:, i:i+1, ...], sf)
            x = self.p(torch.cat((x, ab[:, i+self.n:i+self.n+1, ...].repeat(1, 1, x.size(2), x.size(3))), dim=1))

        return x