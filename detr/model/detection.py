from typing import Dict, List, Tuple, Union
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin
import torch
from torch.nn import functional as F

from detr.model.backbone import BackboneEmbedding
from detr.model.transformer import Transformer
from detr.model.feed_forward_heads import MultilayerPerceptron


class DeTr(DeviceDtypeModuleMixin, torch.nn.Module):
    def __init__(
        self,
        backbone: BackboneEmbedding,
        transformer: Transformer,
        num_classes: int,
        num_queries: int,
        aux_outputs: bool = False,
        ndim: int = 2,
    ) -> None:
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer

        hidden_dim = self.transformer.d_model

        self.class_embed = torch.nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = MultilayerPerceptron(hidden_dim, hidden_dim, 2 * ndim, 3)
        self.query_embed = torch.nn.Embedding(num_queries, hidden_dim)
        self.input_proj = getattr(torch.nn, "Conv%dd" % ndim)(
            backbone.num_channels, hidden_dim, kernel_size=1
        )
        self.backbone = backbone
        self.aux_outputs = aux_outputs

    def forward(
        self, image_batch: torch.Tensor, mask: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]],
    ]:

        feature_tensors, feature_masks, pos = self.backbone(image_batch, mask)

        _src, _mask = feature_tensors[-1], feature_masks[-1]

        hs = self.transformer(
            self.input_proj(_src), _mask, self.query_embed.weight, pos[-1]
        )[0]

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        if self.aux_loss:
            return (
                outputs_class[-1],
                outputs_coord[-1],
                # auxiliary outputs
                [
                    (aux_cls_output, aux_box_output)
                    for aux_cls_output, aux_box_output in zip(outputs_class[:-1], outputs_coord[:-1])
                ],
            )
        return outputs_class[-1], outputs_coord[-1]

