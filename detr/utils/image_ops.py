from typing import List, Tuple
import torch
import torchvision
if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def pad_to_max_size(
    tensor_list: List[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    # determine max img size (including batch dim)
    max_size = torch.stack(
        [
            torch.tensor([len(tensor_list)] + list(_tensor.shape))
            for _tensor in tensor_list
        ]
    ).max(0)[0]

    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    tensor = torch.zeros(max_size, dtype=dtype, device=device)
    mask = torch.ones((max_size[0], *max_size[2:]), dtype=torch.bool, device=device)

    # zero padding to max size and calc padding mask (1 for each padded pixel/voxel)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        if img.ndim == 2:
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
        elif img.ndim == 3:
            pad_img[
                : img.shape[0], : img.shape[1], : img.shape[2], : img.shape[2]
            ].copy_(img)
            m[: img.shape[1], : img.shape[2], : img.shape[3]] = False

    return tensor, mask


def interpolate(
    input, size=None, scale_factor=None, mode="nearest", align_corners=None
):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(
            input, size, scale_factor, mode, align_corners
        )

