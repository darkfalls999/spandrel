from __future__ import annotations

import math

import torch
from typing_extensions import override

from spandrel.architectures.HMA.__arch.HMANet import HMANet

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    SizeRequirements,
    StateDict,
)
from ...util import KeyCondition, get_pixelshuffle_params, get_seq_len


def add_missing_keys(
    state_dict: StateDict,
    # model: HMANet
) -> StateDict:

    missing_keys = [
        "layers.0.residual_group.blocks.9.pre_norm.weight",
        "layers.0.residual_group.blocks.9.pre_norm.bias",
        "layers.0.residual_group.blocks.9.fused_conv.weight",
        "layers.0.residual_group.blocks.9.fused_conv.bias",
        "layers.0.residual_group.blocks.9.norm1.weight",
        "layers.0.residual_group.blocks.9.norm1.bias",
        "layers.0.residual_group.blocks.9.se.fc1.weight",
        "layers.0.residual_group.blocks.9.se.fc1.bias",
        "layers.0.residual_group.blocks.9.se.fc2.weight",
        "layers.0.residual_group.blocks.9.se.fc2.bias",
        "layers.0.residual_group.blocks.9.conv3_1x1.weight",
        "layers.0.residual_group.blocks.9.conv3_1x1.bias",
        "layers.0.residual_group.blocks.10.norm1.weight",
        "layers.0.residual_group.blocks.10.norm1.bias",
        "layers.0.residual_group.blocks.10.qkv.weight",
        "layers.0.residual_group.blocks.10.attn.relative_position_bias_table",
        "layers.0.residual_group.blocks.10.attn.proj.weight",
        "layers.0.residual_group.blocks.10.attn.proj.bias",
        "layers.0.residual_group.blocks.10.norm2.weight",
        "layers.0.residual_group.blocks.10.norm2.bias",
        "layers.0.residual_group.blocks.10.mlp.fc1.weight",
        "layers.0.residual_group.blocks.10.mlp.fc1.bias",
        "layers.0.residual_group.blocks.10.mlp.fc2.weight",
        "layers.0.residual_group.blocks.10.mlp.fc2.bias",
        "layers.0.residual_group.blocks.11.norm1.weight",
        "layers.0.residual_group.blocks.11.norm1.bias",
        "layers.0.residual_group.blocks.11.qkv.weight",
        "layers.0.residual_group.blocks.11.attn.relative_position_bias_table",
        "layers.0.residual_group.blocks.11.attn.proj.weight",
        "layers.0.residual_group.blocks.11.attn.proj.bias",
        "layers.0.residual_group.blocks.11.norm2.weight",
        "layers.0.residual_group.blocks.11.norm2.bias",
        "layers.0.residual_group.blocks.11.mlp.fc1.weight",
        "layers.0.residual_group.blocks.11.mlp.fc1.bias",
        "layers.0.residual_group.blocks.11.mlp.fc2.weight",
        "layers.0.residual_group.blocks.11.mlp.fc2.bias",
        "layers.0.residual_group.blocks.12.pre_norm.weight",
        "layers.0.residual_group.blocks.12.pre_norm.bias",
        "layers.0.residual_group.blocks.12.fused_conv.weight",
        "layers.0.residual_group.blocks.12.fused_conv.bias",
        "layers.0.residual_group.blocks.12.norm1.weight",
        "layers.0.residual_group.blocks.12.norm1.bias",
        "layers.0.residual_group.blocks.12.se.fc1.weight",
        "layers.0.residual_group.blocks.12.se.fc1.bias",
        "layers.0.residual_group.blocks.12.se.fc2.weight",
        "layers.0.residual_group.blocks.12.se.fc2.bias",
        "layers.0.residual_group.blocks.12.conv3_1x1.weight",
        "layers.0.residual_group.blocks.12.conv3_1x1.bias",
        "layers.0.residual_group.blocks.13.norm1.weight",
        "layers.0.residual_group.blocks.13.norm1.bias",
        "layers.0.residual_group.blocks.13.qkv.weight",
        "layers.0.residual_group.blocks.13.attn.relative_position_bias_table",
        "layers.0.residual_group.blocks.13.attn.proj.weight",
        "layers.0.residual_group.blocks.13.attn.proj.bias",
        "layers.0.residual_group.blocks.13.norm2.weight",
        "layers.0.residual_group.blocks.13.norm2.bias",
        "layers.0.residual_group.blocks.13.mlp.fc1.weight",
        "layers.0.residual_group.blocks.13.mlp.fc1.bias",
        "layers.0.residual_group.blocks.13.mlp.fc2.weight",
        "layers.0.residual_group.blocks.13.mlp.fc2.bias",
        "layers.1.residual_group.blocks.9.pre_norm.weight",
        "layers.1.residual_group.blocks.9.pre_norm.bias",
        "layers.1.residual_group.blocks.9.fused_conv.weight",
        "layers.1.residual_group.blocks.9.fused_conv.bias",
        "layers.1.residual_group.blocks.9.norm1.weight",
        "layers.1.residual_group.blocks.9.norm1.bias",
        "layers.1.residual_group.blocks.9.se.fc1.weight",
        "layers.1.residual_group.blocks.9.se.fc1.bias",
        "layers.1.residual_group.blocks.9.se.fc2.weight",
        "layers.1.residual_group.blocks.9.se.fc2.bias",
        "layers.1.residual_group.blocks.9.conv3_1x1.weight",
        "layers.1.residual_group.blocks.9.conv3_1x1.bias",
        "layers.1.residual_group.blocks.10.norm1.weight",
        "layers.1.residual_group.blocks.10.norm1.bias",
        "layers.1.residual_group.blocks.10.qkv.weight",
        "layers.1.residual_group.blocks.10.attn.relative_position_bias_table",
        "layers.1.residual_group.blocks.10.attn.proj.weight",
        "layers.1.residual_group.blocks.10.attn.proj.bias",
        "layers.1.residual_group.blocks.10.norm2.weight",
        "layers.1.residual_group.blocks.10.norm2.bias",
        "layers.1.residual_group.blocks.10.mlp.fc1.weight",
        "layers.1.residual_group.blocks.10.mlp.fc1.bias",
        "layers.1.residual_group.blocks.10.mlp.fc2.weight",
        "layers.1.residual_group.blocks.10.mlp.fc2.bias",
        "layers.1.residual_group.blocks.11.norm1.weight",
        "layers.1.residual_group.blocks.11.norm1.bias",
        "layers.1.residual_group.blocks.11.qkv.weight",
        "layers.1.residual_group.blocks.11.attn.relative_position_bias_table",
        "layers.1.residual_group.blocks.11.attn.proj.weight",
        "layers.1.residual_group.blocks.11.attn.proj.bias",
        "layers.1.residual_group.blocks.11.norm2.weight",
        "layers.1.residual_group.blocks.11.norm2.bias",
        "layers.1.residual_group.blocks.11.mlp.fc1.weight",
        "layers.1.residual_group.blocks.11.mlp.fc1.bias",
        "layers.1.residual_group.blocks.11.mlp.fc2.weight",
        "layers.1.residual_group.blocks.11.mlp.fc2.bias",
        "layers.1.residual_group.blocks.12.pre_norm.weight",
        "layers.1.residual_group.blocks.12.pre_norm.bias",
        "layers.1.residual_group.blocks.12.fused_conv.weight",
        "layers.1.residual_group.blocks.12.fused_conv.bias",
        "layers.1.residual_group.blocks.12.norm1.weight",
        "layers.1.residual_group.blocks.12.norm1.bias",
        "layers.1.residual_group.blocks.12.se.fc1.weight",
        "layers.1.residual_group.blocks.12.se.fc1.bias",
        "layers.1.residual_group.blocks.12.se.fc2.weight",
        "layers.1.residual_group.blocks.12.se.fc2.bias",
        "layers.1.residual_group.blocks.12.conv3_1x1.weight",
        "layers.1.residual_group.blocks.12.conv3_1x1.bias",
        "layers.1.residual_group.blocks.13.norm1.weight",
        "layers.1.residual_group.blocks.13.norm1.bias",
        "layers.1.residual_group.blocks.13.qkv.weight",
        "layers.1.residual_group.blocks.13.attn.relative_position_bias_table",
        "layers.1.residual_group.blocks.13.attn.proj.weight",
        "layers.1.residual_group.blocks.13.attn.proj.bias",
        "layers.1.residual_group.blocks.13.norm2.weight",
        "layers.1.residual_group.blocks.13.norm2.bias",
        "layers.1.residual_group.blocks.13.mlp.fc1.weight",
        "layers.1.residual_group.blocks.13.mlp.fc1.bias",
        "layers.1.residual_group.blocks.13.mlp.fc2.weight",
        "layers.1.residual_group.blocks.13.mlp.fc2.bias",
        "layers.2.residual_group.blocks.9.pre_norm.weight",
        "layers.2.residual_group.blocks.9.pre_norm.bias",
        "layers.2.residual_group.blocks.9.fused_conv.weight",
        "layers.2.residual_group.blocks.9.fused_conv.bias",
        "layers.2.residual_group.blocks.9.norm1.weight",
        "layers.2.residual_group.blocks.9.norm1.bias",
        "layers.2.residual_group.blocks.9.se.fc1.weight",
        "layers.2.residual_group.blocks.9.se.fc1.bias",
        "layers.2.residual_group.blocks.9.se.fc2.weight",
        "layers.2.residual_group.blocks.9.se.fc2.bias",
        "layers.2.residual_group.blocks.9.conv3_1x1.weight",
        "layers.2.residual_group.blocks.9.conv3_1x1.bias",
        "layers.2.residual_group.blocks.10.norm1.weight",
        "layers.2.residual_group.blocks.10.norm1.bias",
        "layers.2.residual_group.blocks.10.qkv.weight",
        "layers.2.residual_group.blocks.10.attn.relative_position_bias_table",
        "layers.2.residual_group.blocks.10.attn.proj.weight",
        "layers.2.residual_group.blocks.10.attn.proj.bias",
        "layers.2.residual_group.blocks.10.norm2.weight",
        "layers.2.residual_group.blocks.10.norm2.bias",
        "layers.2.residual_group.blocks.10.mlp.fc1.weight",
        "layers.2.residual_group.blocks.10.mlp.fc1.bias",
        "layers.2.residual_group.blocks.10.mlp.fc2.weight",
        "layers.2.residual_group.blocks.10.mlp.fc2.bias",
        "layers.2.residual_group.blocks.11.norm1.weight",
        "layers.2.residual_group.blocks.11.norm1.bias",
        "layers.2.residual_group.blocks.11.qkv.weight",
        "layers.2.residual_group.blocks.11.attn.relative_position_bias_table",
        "layers.2.residual_group.blocks.11.attn.proj.weight",
        "layers.2.residual_group.blocks.11.attn.proj.bias",
        "layers.2.residual_group.blocks.11.norm2.weight",
        "layers.2.residual_group.blocks.11.norm2.bias",
        "layers.2.residual_group.blocks.11.mlp.fc1.weight",
        "layers.2.residual_group.blocks.11.mlp.fc1.bias",
        "layers.2.residual_group.blocks.11.mlp.fc2.weight",
        "layers.2.residual_group.blocks.11.mlp.fc2.bias",
        "layers.2.residual_group.blocks.12.pre_norm.weight",
        "layers.2.residual_group.blocks.12.pre_norm.bias",
        "layers.2.residual_group.blocks.12.fused_conv.weight",
        "layers.2.residual_group.blocks.12.fused_conv.bias",
        "layers.2.residual_group.blocks.12.norm1.weight",
        "layers.2.residual_group.blocks.12.norm1.bias",
        "layers.2.residual_group.blocks.12.se.fc1.weight",
        "layers.2.residual_group.blocks.12.se.fc1.bias",
        "layers.2.residual_group.blocks.12.se.fc2.weight",
        "layers.2.residual_group.blocks.12.se.fc2.bias",
        "layers.2.residual_group.blocks.12.conv3_1x1.weight",
        "layers.2.residual_group.blocks.12.conv3_1x1.bias",
        "layers.2.residual_group.blocks.13.norm1.weight",
        "layers.2.residual_group.blocks.13.norm1.bias",
        "layers.2.residual_group.blocks.13.qkv.weight",
        "layers.2.residual_group.blocks.13.attn.relative_position_bias_table",
        "layers.2.residual_group.blocks.13.attn.proj.weight",
        "layers.2.residual_group.blocks.13.attn.proj.bias",
        "layers.2.residual_group.blocks.13.norm2.weight",
        "layers.2.residual_group.blocks.13.norm2.bias",
        "layers.2.residual_group.blocks.13.mlp.fc1.weight",
        "layers.2.residual_group.blocks.13.mlp.fc1.bias",
        "layers.2.residual_group.blocks.13.mlp.fc2.weight",
        "layers.2.residual_group.blocks.13.mlp.fc2.bias",
        "layers.3.residual_group.blocks.10.attn.relative_position_bias_table",
        "layers.3.residual_group.blocks.11.attn.relative_position_bias_table",
        "layers.3.residual_group.blocks.13.attn.relative_position_bias_table",
        "layers.4.residual_group.blocks.10.attn.relative_position_bias_table",
        "layers.4.residual_group.blocks.11.attn.relative_position_bias_table",
        "layers.4.residual_group.blocks.13.attn.relative_position_bias_table",
        "layers.5.residual_group.blocks.9.pre_norm.weight",
        "layers.5.residual_group.blocks.9.pre_norm.bias",
        "layers.5.residual_group.blocks.9.fused_conv.weight",
        "layers.5.residual_group.blocks.9.fused_conv.bias",
        "layers.5.residual_group.blocks.9.norm1.weight",
        "layers.5.residual_group.blocks.9.norm1.bias",
        "layers.5.residual_group.blocks.9.se.fc1.weight",
        "layers.5.residual_group.blocks.9.se.fc1.bias",
        "layers.5.residual_group.blocks.9.se.fc2.weight",
        "layers.5.residual_group.blocks.9.se.fc2.bias",
        "layers.5.residual_group.blocks.9.conv3_1x1.weight",
        "layers.5.residual_group.blocks.9.conv3_1x1.bias",
        "layers.5.residual_group.blocks.10.norm1.weight",
        "layers.5.residual_group.blocks.10.norm1.bias",
        "layers.5.residual_group.blocks.10.qkv.weight",
        "layers.5.residual_group.blocks.10.attn.relative_position_bias_table",
        "layers.5.residual_group.blocks.10.attn.proj.weight",
        "layers.5.residual_group.blocks.10.attn.proj.bias",
        "layers.5.residual_group.blocks.10.norm2.weight",
        "layers.5.residual_group.blocks.10.norm2.bias",
        "layers.5.residual_group.blocks.10.mlp.fc1.weight",
        "layers.5.residual_group.blocks.10.mlp.fc1.bias",
        "layers.5.residual_group.blocks.10.mlp.fc2.weight",
        "layers.5.residual_group.blocks.10.mlp.fc2.bias",
        "layers.5.residual_group.blocks.11.norm1.weight",
        "layers.5.residual_group.blocks.11.norm1.bias",
        "layers.5.residual_group.blocks.11.qkv.weight",
        "layers.5.residual_group.blocks.11.attn.relative_position_bias_table",
        "layers.5.residual_group.blocks.11.attn.proj.weight",
        "layers.5.residual_group.blocks.11.attn.proj.bias",
        "layers.5.residual_group.blocks.11.norm2.weight",
        "layers.5.residual_group.blocks.11.norm2.bias",
        "layers.5.residual_group.blocks.11.mlp.fc1.weight",
        "layers.5.residual_group.blocks.11.mlp.fc1.bias",
        "layers.5.residual_group.blocks.11.mlp.fc2.weight",
        "layers.5.residual_group.blocks.11.mlp.fc2.bias",
        "layers.5.residual_group.blocks.12.pre_norm.weight",
        "layers.5.residual_group.blocks.12.pre_norm.bias",
        "layers.5.residual_group.blocks.12.fused_conv.weight",
        "layers.5.residual_group.blocks.12.fused_conv.bias",
        "layers.5.residual_group.blocks.12.norm1.weight",
        "layers.5.residual_group.blocks.12.norm1.bias",
        "layers.5.residual_group.blocks.12.se.fc1.weight",
        "layers.5.residual_group.blocks.12.se.fc1.bias",
        "layers.5.residual_group.blocks.12.se.fc2.weight",
        "layers.5.residual_group.blocks.12.se.fc2.bias",
        "layers.5.residual_group.blocks.12.conv3_1x1.weight",
        "layers.5.residual_group.blocks.12.conv3_1x1.bias",
        "layers.5.residual_group.blocks.13.norm1.weight",
        "layers.5.residual_group.blocks.13.norm1.bias",
        "layers.5.residual_group.blocks.13.qkv.weight",
        "layers.5.residual_group.blocks.13.attn.relative_position_bias_table",
        "layers.5.residual_group.blocks.13.attn.proj.weight",
        "layers.5.residual_group.blocks.13.attn.proj.bias",
        "layers.5.residual_group.blocks.13.norm2.weight",
        "layers.5.residual_group.blocks.13.norm2.bias",
        "layers.5.residual_group.blocks.13.mlp.fc1.weight",
        "layers.5.residual_group.blocks.13.mlp.fc1.bias",
        "layers.5.residual_group.blocks.13.mlp.fc2.weight",
        "layers.5.residual_group.blocks.13.mlp.fc2.bias",
    ]

    # model_state_dict = model.state_dict()

    for key in missing_keys:
        # if key in model_state_dict:
        #    if key not in state_dict or state_dict[key].shape != model_state_dict[key].shape:
        #        # Handle weight and bias initialization for size mismatch
        #        if key.endswith("weight"):
        #            state_dict[key] = torch.randn_like(model_state_dict[key])
        #        elif key.endswith("bias"):
        #            state_dict[key] = torch.zeros_like(model_state_dict[key])
        # else:
        #    # Add missing keys if they do not exist in state_dict
        #    if key.endswith("weight"):
        #        state_dict[key] = torch.randn(1)  # Initialize with random values for weights
        #    elif key.endswith("bias"):
        #        state_dict[key] = torch.zeros(1)  # Initialize with zeros for biases

        # Add missing keys if they do not exist in state_dict
        if key.endswith("weight"):
            state_dict[key] = torch.randn(
                1
            )  # Initialize with random values for weights
        elif key.endswith("bias"):
            state_dict[key] = torch.zeros(1)  # Initialize with zeros for biases

    return state_dict


class HMANetArch(Architecture[HMANet]):
    def __init__(self) -> None:
        super().__init__(
            id="HMANet",
            detect=KeyCondition.has_all(
                "layers.2.residual_group.gab.qkv.bias",
                "layers.2.residual_group.gab.qkv.weight",
                "layers.3.residual_group.blocks.0.conv3_1x1.bias",
                "layers.3.residual_group.blocks.0.conv3_1x1.weight",
                "layers.4.residual_group.gab.grid_attn.attn_transform1.pos.pos1.0.bias",
                "layers.5.residual_group.gab.window_attn_s.proj.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[HMANet]:
        img_size = 64
        patch_size = 1
        in_chans = 3
        embed_dim = 96
        depths = (6, 6, 6, 6)
        num_heads = (6, 6, 6, 6)
        window_size = 7
        conv_scale = 0.01  # cannot be deduced from state dict
        mlp_ratio = 2.0
        qkv_bias = True
        qk_scale = None  # cannot be deduced from state dict
        drop_rate = 0.0  # cannot be deduced from state dict
        attn_drop_rate = 0.0  # cannot be deduced from state dict
        drop_path_rate = 0.1  # cannot be deduced from state dict
        ape = False
        patch_norm = True
        upscale = 2
        img_range = 1.0  # cannot be deduced from state dict
        upsampler = "pixelshuffle"  # it's the only possible value
        resi_connection = "1conv"
        use_checkpoint = True

        in_chans = state_dict["conv_first.weight"].shape[1]

        upscale, num_feat = get_pixelshuffle_params(state_dict, "upsample")

        window_size = int(math.sqrt(state_dict["relative_position_index_SA"].shape[0]))

        embed_dim = state_dict["conv_first.weight"].shape[0]

        mlp_ratio = float(
            state_dict["layers.0.residual_group.blocks.1.mlp.fc1.bias"].shape[0]
            / embed_dim
        )

        # num_layers = len(depths)
        num_layers = get_seq_len(state_dict, "layers")
        depths = []
        num_heads = []
        for i in range(num_layers):
            depths.append(
                get_seq_len(state_dict, f"layers.{i}.residual_group.blocks") - 3
            )
            num_heads.append(
                state_dict[
                    f"layers.{i}.residual_group.blocks.1.attn.relative_position_bias_table"
                ].shape[1]
            )

        if "conv_after_body.weight" in state_dict:
            resi_connection = "1conv"
        else:
            resi_connection = "3conv"

        # qkv_bias = "layers.0.residual_group.blocks.0.attn.qkv.bias" in state_dict
        patch_norm = "patch_embed.norm.weight" in state_dict
        ape = "absolute_pos_embed" in state_dict

        # The JPEG models are the only ones with window-size 7, and they also use this range
        img_range = 255.0 if window_size == 7 else 1.0

        model = HMANet(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            conv_scale=conv_scale,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            ape=ape,
            patch_norm=patch_norm,
            upscale=upscale,
            img_range=img_range,
            upsampler=upsampler,
            resi_connection=resi_connection,
            use_checkpoint=use_checkpoint,
            num_feat=num_feat,
        )

        def call(model: HMANet, image: torch.Tensor) -> torch.Tensor:
            """
            It will first crop input images to tiles, and then process each tile.
            Finally, all the processed tiles are merged into one images.

            Modified from: https://github.com/ata4/esrgan-launcher

            Reference: https://github.com/korouuuuu/HMA/blob/main/hma/models/hma_model.py
            """

            scale = upscale
            batch, channel, height, width = image.shape
            output_height = height * scale
            output_width = width * scale
            output_shape = (batch, channel, output_height, output_width)
            tile_size = int(state_dict["upsample.0.weight"].shape[0])
            tile_pad = int(state_dict["upsample.0.weight"].shape[1] / 2)

            # start with black image
            output = image.new_zeros(output_shape)
            tiles_w = math.ceil(width / tile_size if width >= tile_size else width)
            tiles_h = math.ceil(height / tile_size if height >= tile_size else height)

            # loop over all tiles
            for h in range(tiles_h):
                for w in range(tiles_w):
                    # extract tile from input image
                    ofs_x = w * tile_size
                    ofs_y = h * tile_size
                    # input tile area on total image
                    input_start_x = ofs_x
                    input_end_x = min(ofs_x + tile_size, width)
                    input_start_y = ofs_y
                    input_end_y = min(ofs_y + tile_size, height)

                    # input tile area on total image with padding
                    input_start_x_pad = max(input_start_x - tile_pad, 0)
                    input_end_x_pad = min(input_end_x + tile_pad, width)
                    input_start_y_pad = max(input_start_y - tile_pad, 0)
                    input_end_y_pad = min(input_end_y + tile_pad, height)

                    # input tile dimensions
                    input_tile_width = input_end_x - input_start_x
                    input_tile_height = input_end_y - input_start_y
                    # tile_idx = h * tiles_w + w + 1
                    input_tile = image[
                        :,
                        :,
                        input_start_y_pad:input_end_y_pad,
                        input_start_x_pad:input_end_x_pad,
                    ]

                    # upscale tile
                    with torch.no_grad():
                        output_tile = model(input_tile)

                    # print(f'\tTile {tile_idx}/{tiles_w * tiles_h}')

                    # output tile area on total image
                    output_start_x = input_start_x * upscale
                    output_end_x = input_end_x * upscale
                    output_start_y = input_start_y * upscale
                    output_end_y = input_end_y * upscale

                    # output tile area without padding
                    output_start_x_tile = (input_start_x - input_start_x_pad) * upscale
                    output_end_x_tile = output_start_x_tile + input_tile_width * upscale
                    output_start_y_tile = (input_start_y - input_start_y_pad) * upscale
                    output_end_y_tile = (
                        output_start_y_tile + input_tile_height * upscale
                    )

                    # put tile into output image
                    output[
                        :, :, output_start_y:output_end_y, output_start_x:output_end_x
                    ] = output_tile[
                        :,
                        :,
                        output_start_y_tile:output_end_y_tile,
                        output_start_x_tile:output_end_x_tile,
                    ]

            # print(f'\tTile {tiles_w * tiles_h}/{tiles_w * tiles_h}')

            return output

        head_length = len(depths)  # type: ignore
        if head_length <= 4:
            size_tag = "small"
        elif head_length < 9:
            size_tag = "medium"
        else:
            size_tag = "large"

        tags = [
            size_tag,
            f"s{img_size}w{window_size}",
            f"{num_feat}nf",
            f"{embed_dim}dim",
            f"{resi_connection}",
        ]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration" if upscale == 1 else "SR",
            tags=tags,
            supports_half=False,  # Too much weirdness to support this at the moment
            supports_bfloat16=True,
            scale=upscale,
            input_channels=in_chans,
            output_channels=in_chans,
            size_requirements=SizeRequirements(minimum=16, multiple_of=16),
            call_fn=call,
        )


__all__ = ["HMANetArch", "HMANet"]
