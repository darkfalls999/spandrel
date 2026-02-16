import torch
from typing_extensions import override

from spandrel.util import KeyCondition, get_seq_len

from ...__helpers.model_descriptor import (
    Architecture,
    ImageModelDescriptor,
    # SizeRequirements,
    StateDict,
)
from ...__helpers.size_req import SizeRequirements
from .__arch.MFDIN_arch import MFDIN_OLD2P


class MFDINArch(Architecture[MFDIN_OLD2P]):
    def __init__(self) -> None:
        super().__init__(
            id="MFDIN",
            detect=KeyCondition.has_all(
                "conv_first.weight",
                "feature_extraction.0.conv1.weight",
                "AlignMoudle.cas_offset_conv1.weight",
                "recon_trunk.0.res1.conv1.weight",
            ),
        )

    @override
    def load(self, state_dict: StateDict) -> ImageModelDescriptor[MFDIN_OLD2P]:

        upscale = 1
        input_channels = 3
        output_channels = 3
        nf = 64  # Number of filters
        groups = 4  # Number of groups in convolutions
        front_rbs = 5  # Number of residual blocks in the front
        back_rfas = 2  # Number of residual feature aggregation blocks in the back
        nfields = 5  # Hyperparameter specific to the model architecture

        nf = state_dict["conv_first.weight"].shape[0]

        front_rbs = get_seq_len(state_dict, "feature_extraction")

        back_rfas = get_seq_len(state_dict, "recon_trunk")

        model = MFDIN_OLD2P(
            nf=nf,
            groups=groups,
            front_RBs=front_rbs,
            back_RFAs=back_rfas,
            nfields=nfields,
        )

        def call(model: MFDIN_OLD2P, image: torch.Tensor) -> torch.Tensor:

            device = image.device

            _, _, H, W = image.size()

            # pad_w = math.ceil(W / 8) * 8
            # pad_h = math.ceil(H / 8) * 8
            # padding = (0, pad_w - W, 0, pad_h - H, 0, 0)
            # need_pad = any(p > 0 for p in padding)

            if image.dim() == 4:  # (B, C, H, W)
                image = image.unsqueeze(1).to(device)  # (B, 1, C, H, W)

            N = 3  # Example value for N
            image = image.repeat(1, N, 1, 1, 1).to(device)

            # if need_pad:
            # image = F.pad(image, padding, "replicate").to(device)

            output = model(image).to(device)

            # if need_pad:
            # output = output[:, :, : H, : W]

            B, N, C, H, W = output.size()

            output = output.view(B, N * C, H, W).to(device)

            return output

        tags = [
            f"{nf}nf",
        ]

        return ImageModelDescriptor(
            model,
            state_dict,
            architecture=self,
            purpose="Restoration",
            tags=tags,
            supports_half=True,  # Too much weirdness to support this at the moment
            supports_bfloat16=True,
            scale=upscale,
            input_channels=input_channels,
            output_channels=output_channels,
            size_requirements=SizeRequirements(multiple_of=8),
            call_fn=call,
        )


__all__ = ["MFDINArch", "MFDIN_OLD2P"]
