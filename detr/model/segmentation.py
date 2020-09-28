from typing import List, Tuple, Union
import torch

from detr.model.transformer import Transformer
from detr.model.backbone import BackboneEmbedding
from detr.model.detection import DeTr
from detr.model.feed_forward_heads import MHAttentionMap, MaskHeadSmallConv


class DeTrSegm(DeTr):
    def __init__(self,
        backbone: BackboneEmbedding,
        transformer: Transformer,
        num_classes: int,
        num_queries: int,
        aux_outputs: bool = False,
        ndim: int = 2,
        freeze_detection: bool = False
    ):
        super().__init__(backbone, transformer, num_classes, num_queries, aux_outputs, ndim)

        if freeze_detection:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = self.transformer.d_model, self.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def forward(
        self, image_batch: torch.Tensor, mask: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]],
    ]:

        feature_tensors, feature_masks, pos = self.backbone(image_batch, mask)

        _src, _mask = feature_tensors[-1], feature_masks[-1]
        _src_proj = self.input_proj(_src)

        hs, memory = self.transformer(
            _src_proj, _mask, self.query_embed.weight, pos[-1]
        )

        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        # h_boxes takes the last one computed
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=_mask)
        seg_masks = self.mask_head(_src_proj, bbox_mask, [feature_tensors[2], feature_tensors[1], feature_tensors[0]])

        outputs_seg_masks = seg_masks.view(feature_tensors[-1].size(0), self.num_queries, *seg_masks.shape[2:])

        if self.aux_loss:
            return (
                outputs_class[-1],
                outputs_coord[-1],
                outputs_seg_masks,
                # auxiliary outputs
                [
                    (aux_cls_output, aux_box_output)
                    for aux_cls_output, aux_box_output in zip(outputs_class[:-1], outputs_coord[:-1])
                ],
            )
        return outputs_class[-1], outputs_coord[-1], outputs_seg_masks
        