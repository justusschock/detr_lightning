import torch


def expand(tensor: torch.Tensor, length: int) -> torch.Tensor:
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)
