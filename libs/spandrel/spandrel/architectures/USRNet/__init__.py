from __future__ import annotations

import numpy as np
import torch
from typing_extensions import override

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from ...util import KeyCondition
from ..__arch_helpers import utils_deblur
from ..__arch_helpers import utils_image as util
from ..__arch_helpers import utils_sisr as sr
from .__arch.network_usrnet import USRNet


class USRNetArch(Architecture[USRNet]):
    def __init__(self) -> None:
        super().__init__(
            id="USRNet",
            detect=KeyCondition.has_all(
                "p.m_head.weight",
                "p.m_down1.0.res.0.weight",
                "p.m_down1.0.res.2.weight",
                "p.m_down2.0.res.0.weight",
                "p.m_down2.0.res.2.weight",
                "p.m_down3.0.res.0.weight",
                "p.m_down3.0.res.2.weight",
                "p.m_body.0.res.0.weight",
                "p.m_body.0.res.2.weight",
                "p.m_tail.weight",
                "h.mlp.0.weight",
                "h.mlp.0.bias",
                "h.mlp.2.weight",
                "h.mlp.2.bias",
                "h.mlp.4.weight",
                "h.mlp.4.bias",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[USRNet]:

        # default values
        # n_iter: int = 8    # n_iter=6 for tiny
        # h_nc: int = 64
        # in_nc = 4
        # out_nc = 3
        # nc = [64, 128, 256, 512]   # [16, 32, 64, 64] for tiny
        # nb = 2

        n_iter = state_dict["h.mlp.4.weight"].shape[0] // 2
        h_nc = state_dict["h.mlp.0.weight"].shape[0]

        in_nc = state_dict["p.m_head.weight"].shape[1]
        out_nc = state_dict["p.m_tail.weight"].shape[0]

        nc = [
            state_dict["p.m_down1.0.res.0.weight"].shape[0],
            state_dict["p.m_down2.0.res.0.weight"].shape[0],
            state_dict["p.m_down3.0.res.0.weight"].shape[0],
            state_dict["p.m_body.0.res.0.weight"].shape[0],
        ]

        act_mode: str = "R"

        # Model variants
        if nc[3] < 512:  # Tiny variants have smaller channel sizes
            model_type = "tiny"
        else:
            model_type = "standard"

        model = USRNet(
            n_iter=n_iter,
            h_nc=h_nc,
            in_nc=in_nc,
            out_nc=out_nc,
            nc=nc,
            act_mode=act_mode,
        )

        def call(model: USRNet, image: torch.Tensor) -> torch.Tensor:
            """
            Reference: https://github.com/cszn/USRNet/blob/master/main_test_realapplication.py
            """
            device = image.device

            scale_factor = 1

            noise_level_img = 3.0  # noise level for LR image, 0.5~3 for clean images
            noise_level_model = noise_level_img / 255.0  # noise level of model

            kernel_width_default_x1234 = [
                0.4,
                0.6,
                1.4,
                1.6,
            ]  # default Gaussian kernel widths of clean/sharp images for x1[0.4], x2[0.7], x3[1.5], x4[2.0]
            kernel_width = kernel_width_default_x1234[scale_factor - 1]

            k = utils_deblur.fspecial("gaussian", 25, kernel_width)
            k = torch.from_numpy(k).float()
            k = sr.shift_pixel(k, scale_factor)  # shift the kernel
            k = k.numpy()
            k /= np.sum(k)
            kernel = util.single2tensor4(k[..., np.newaxis]).to(device)

            sigma = torch.tensor(noise_level_model).view([1, 1, 1, 1]).to(device)

            return model(image, kernel, scale_factor, sigma).to(device)

        tags = [f"{len(nc)}nc, {n_iter}niter, {model_type}"]

        upscale = 1

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=tags,
            supports_half=False,
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_nc,
            output_channels=out_nc,
            size_requirements=SizeRequirements(minimum=16, multiple_of=8),
            call_fn=call,
        )


__all__ = ["USRNetArch", "USRNet"]
