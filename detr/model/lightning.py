from detr.model.matcher import HungarianMatcher
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from pytorch_lightning import LightningModule

from torch.optim.optimizer import Optimizer
from detr.model.embedding import PositionEmbeddingLearned, PositionEmbeddingSine
from detr.model.backbone import BackboneEmbedding, ResNetBackbone
from detr.model.transformer import Transformer
from detr.model.detection import Detr as _Detr
from detr.model.segmentation import DetrSegm as _DetrSegm
from detr.losses.detection import DetectionLoss
from detr.losses.segmentation import SegmentationLoss

__all__ = ["Detr"]


class Detr(LightningModule):
    def __init__(
        self,
        num_classes: int,
        num_queries: int = 100,
        backbone_name: str = "resnet50",
        train_backbone: bool = True,
        return_interm_layers_backbone: bool = True,
        dilation_backbone: bool = False,
        pretrained_backbone: bool = True,
        num_pos_feats: int = 256,
        learned_pos_embedding: bool = False,
        d_model_transformer: int = 512,
        nhead_transformer: int = 8,
        num_encoder_layers_transformer: int = 6,
        num_decoder_layers_transformer: int = 6,
        dim_feedforward_transformer: int = 2048,
        dropout_transformer: float = 0.1,
        activation_transformer: str = "relu",
        normalize_before_transformer: bool = False,
        return_intermediate_dec_transformer: bool = False,
        aux_outputs: bool = False,
        ndim: int = 2,
        freeze_detection: bool = False,
        segmentation: bool = False,
        pos_embedding_kwargs: Optional[dict] = None,
        lr: float = 1e-4,
        lr_backbone: float = 1e-5,
        weight_decay: float = 1e-4,
        lr_decay_step: int = 200,
        cost_class_hungarian_matcher: float = 1,
        cost_bbox_hungarian_matcher: float = 5,
        cost_giou_hungarian_matcher: float = 2,
        weight_dict_loss: Optional[dict] = None,
        losses: Sequence = ("labels", "boxes", "cardinality"),
        eos_coeff_loss: float = 0.1,
    ) -> None:

        super().__init__()

        self.save_hyperparameters()
        self.segmentation = segmentation
        self.aux_outputs = aux_outputs

        # build backbone CNN
        backbone = ResNetBackbone(
            name=backbone_name,
            train_backbone=train_backbone,
            return_interm_layers=return_interm_layers_backbone,
            dilation=dilation_backbone,
            pretrained=pretrained_backbone,
        )

        # Build Position Embedding
        if pos_embedding_kwargs is None:
            pos_embedding_kwargs = {}
        if learned_pos_embedding:
            position_embedding = PositionEmbeddingLearned(
                num_pos_feats=num_pos_feats, ndim=ndim
            )
        else:
            assert ndim == 2
            position_embedding = PositionEmbeddingSine(
                num_pos_feats=num_pos_feats, ndim=ndim, **pos_embedding_kwargs
            )

        # Build Combined Backbone and Embedding
        backbone_embedding = BackboneEmbedding(
            backbone=backbone, position_embedding=position_embedding
        )

        # build transformer
        transformer = Transformer(
            d_model=d_model_transformer,
            nhead=nhead_transformer,
            num_encoder_layers=num_encoder_layers_transformer,
            num_decoder_layers=num_decoder_layers_transformer,
            dim_feedforward=dim_feedforward_transformer,
            dropout=dropout_transformer,
            activation=activation_transformer,
            normalize_before=normalize_before_transformer,
            return_intermediate_dec=return_intermediate_dec_transformer,
        )

        # build hungarian matcher for loss calc
        matcher = HungarianMatcher(
            cost_class=cost_class_hungarian_matcher,
            cost_bbox=cost_bbox_hungarian_matcher,
            cost_giou=cost_giou_hungarian_matcher,
            ndim=ndim,
        )

        if weight_dict_loss is None:
            weight_dict_loss = {"loss_ce": 1, "loss_bbox": 5}
            if aux_outputs:
                aux_weight_dict = {}
                for i in range(num_decoder_layers_transformer - 1):
                    aux_weight_dict.update(
                        {k + f"_{i}": v for k, v in weight_dict_loss.items()}
                    )
                weight_dict_loss.update(aux_weight_dict)
        if segmentation:
            model = _DetrSegm(
                backbone_embedding,
                transformer,
                num_classes,
                num_queries,
                aux_outputs,
                ndim,
                freeze_detection,
            )
            if "masks" not in losses:
                losses = (*losses, "masks")
            if "loss_mask" not in weight_dict_loss:
                weight_dict_loss.update(loss_mask=1)

            if "loss_dice" not in weight_dict_loss:
                weight_dict_loss.update(loss_dice=1)
            loss = SegmentationLoss(
                num_classes, matcher, weight_dict_loss, eos_coeff_loss, losses
            )
        else:
            model = _Detr(
                backbone_embedding,
                transformer,
                num_classes,
                num_queries,
                aux_outputs,
                ndim,
            )
            loss = DetectionLoss(
                num_classes, matcher, weight_dict_loss, eos_coeff_loss, losses
            )

        self.model = model
        self.loss_calculator = loss

    def configure_optimizers(
        self,
    ) -> Optional[
        Union[Optimizer, Sequence[Optimizer], Dict, Sequence[Dict], Tuple[List, List]]
    ]:

        param_dicts = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone.backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if "backbone.backbone" in n and p.requires_grad
                ],
                "lr": self.hparams["lr_backbone"],
            },
        ]

        optim = torch.optim.AdamW(
            param_dicts,
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        return (
            [optim],
            [torch.optim.lr_scheduler.StepLR(optim, self.hparams["lr_decay_step"])],
        )

    def forward(
        self, image_batch: torch.Tensor, padding_mask: Optional[torch.Tensor] = None
    ):
        if padding_mask is None:
            padding_mask = torch.zeros(
                (
                    image_batch.size(0),
                    *image_batch.shape[2:],
                ),
                device=image_batch.device,
                dtype=torch.long,
            )
        return self.model(image_batch, padding_mask)

    def training_step(
        self, batch: dict, batch_idx: int
    ) -> Union[int, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]:

        outputs = self.model(batch["data"], batch["padding_mask"])
        targets = batch["label"]

        for t in targets:
            # targets must be dict with boxes (key: boxes), class labels (key: labels) and optionally masks (key: masks)
            assert "boxes" in t
            assert "labels" in t

            if isinstance(self.loss_calculator, SegmentationLoss):
                assert "masks" in t

        if self.segmentation:
            if self.aux_outputs:
                outputs_class, outputs_boxes, outputs_masks, aux_outputs = outputs
            else:
                outputs_class, outputs_boxes, outputs_masks = outputs
                aux_outputs = None

        else:
            outputs_masks = None
            if self.aux_outputs:
                outputs_class, outputs_boxes, aux_outputs = outputs
            else:
                outputs_class, outputs_boxes = outputs
                aux_outputs = None

        losses = self.loss_calculator(
            outputs_class=outputs_class,
            outputs_boxes=outputs_boxes,
            targets=targets,
            outputs_masks=outputs_masks,
            aux_outputs=aux_outputs,
        )

        for k, v in losses.items():
            self.logger.experiment.add_scalar(f"train/{k}", v)

        with self.loss_calculator.combine_losses(losses) as total_loss:
            return {"loss": total_loss, **losses}
