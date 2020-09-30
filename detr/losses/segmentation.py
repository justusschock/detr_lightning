from detr.utils.image_ops import pad_to_max_size, interpolate
from typing import Dict, List, Optional

import torch
from torch.nn import functional as F
from detr.losses.detection import DetectionLoss

__all__ = ["SegmentationLoss", "dice_loss", "sigmoid_focal_loss"]


class SegmentationLoss(DetectionLoss):
    def loss_masks(
        self,
        outputs_class: torch.Tensor,
        outputs_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
        indices,
        num_boxes,
        outputs_masks,
        **kwargs
    ):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs_masks

        target_masks, _ = pad_to_max_size([t["masks"] for t in targets])
        target_masks = target_masks.to(src_masks)

        src_masks = src_masks[src_idx]
        # upsample predictions to the target size
        if src_masks.ndim - 2 == 2:
            interp_mode = "bilinear"
        elif src_masks.ndim - 2 == 3:
            interp_mode = "trilinear"
        else:
            raise RuntimeError

        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[2:],
            mode=interp_mode,
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks[tgt_idx].flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def get_loss(
        self,
        loss,
        outputs_class,
        outputs_boxes,
        targets,
        indices,
        num_boxes,
        output_masks: Optional[torch.Tensor] = None,
        **kwargs
    ):

        if loss == "masks":
            return self.loss_masks(
                outputs_class,
                outputs_boxes,
                targets,
                indices,
                num_boxes,
                output_masks,
                **kwargs
            )

        return super().get_loss(
            loss,
            outputs_class,
            outputs_boxes,
            targets,
            indices,
            num_boxes,
            output_masks,
            **kwargs
        )


def dice_loss(
    inputs: torch.Tensor, targets: torch.Tensor, num_boxes: int
) -> torch.Tensor:
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_boxes: int,
    alpha: float = 0.25,
    gamma: float = 2,
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes
