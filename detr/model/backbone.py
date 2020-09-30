from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
from torch.nn import functional as F
import torchvision

from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin

from detr.model.embedding import PositionEmbedding

__all__ = ["FrozenBatchNorm", "BackboneBase", "ResNetBackbone", "BackboneEmbedding"]


class FrozenBatchNorm(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rsqrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List,
        unexpected_keys: List,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reshape = [1, -1] + ([1] * (x.ndim - 2))
        w = self.weight.reshape(*reshape)
        b = self.bias.reshape(*reshape)
        rv = self.running_var.reshape(*reshape)
        rm = self.running_mean.reshape(*reshape)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(torch.nn.Module):
    def __init__(
        self,
        backbone: torch.nn.Module,
        train_backbone: bool,
        num_channels: int,
        return_interm_layers: bool,
    ) -> None:
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if (
                not train_backbone
                or "layer2" not in name
                and "layer3" not in name
                and "layer4" not in name
            ):
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = torchvision.models._utils.IntermediateLayerGetter(
            backbone, return_layers=return_layers
        )
        self.num_channels = num_channels

    def forward(
        self, image_batch: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        xs: Dict[str, torch.Tensor] = self.body(image_batch)

        out: Dict[str, Dict[str, torch.Tensor]] = OrderedDict({})

        for name, x in xs.items():
            mask_: torch.Tensor = F.interpolate(
                mask[:, None].float(), size=x.shape[-(image_batch.ndim - 2) :]
            ).to(torch.bool)[:, 0]
            out[name] = {"tensor": x, "mask": mask_}

        return out


class ResNetBackbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(
        self,
        name: str,
        train_backbone: bool,
        return_interm_layers: bool,
        dilation: bool,
        pretrained: bool = True,
    ) -> None:
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained,
            norm_layer=FrozenBatchNorm,
        )
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class BackboneEmbedding(DeviceDtypeModuleMixin, torch.nn.Module):
    def __init__(
        self, backbone: BackboneBase, position_embedding: PositionEmbedding
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.position_embedding = position_embedding

    def forward(
        self, image_batch: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        xs = self.backbone(image_batch, mask)

        out_tensors = []
        out_masks = []
        pos = []

        for _, x in xs.items():
            # x is now a dict with keys 'mask' and 'tensor'
            out_tensors.append(x["tensor"])
            out_masks.append(x["mask"])
            pos.append(self.position_embedding(x["tensor"], x["mask"]))

        return out_tensors, out_masks, pos
