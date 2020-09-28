import math
from typing import Optional
from abc import ABC, abstractmethod
import torch
from pytorch_lightning.utilities.device_dtype_mixin import DeviceDtypeModuleMixin


class PositionEmbedding(DeviceDtypeModuleMixin, torch.nn.Module, ABC):
    def __init__(self, num_pos_feats: int, ndim: int):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.ndim = ndim

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def reset_parameters(self) -> None:
        pass


class PositionEmbeddingSine(PositionEmbedding):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False, scale: Optional[float]=None, ndim: int = 2):
        super().__init__(num_pos_feats, ndim)

        assert self.ndim == 2
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def reset_parameters(self) -> None:
        pass

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(PositionEmbedding):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats: int = 256, ndim: int = 3):
        super().__init__(num_pos_feats, ndim)

        for idx in range(self.ndim):
            setattr(self, "embed_dim_%d" % idx, torch.nn.Embedding(50, self.num_pos_feats))
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(self.ndim):
            torch.nn.init.uniform(getattr(self, "embed_dim_%d" % idx))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # add mask for interface consistency between learned embedding and other

        embeddings = []

        for idx in range(self.ndim):
            i = torch.arange(x.shape[-self.ndim], device=x.device)
            emb = getattr(self, "embed_dim_%d" % idx)(i)
            target_shape = x.shape[-self.ndim :]
            target_shape[-(idx + 1)] = 1
            target_shape += [1]
            view_shape = [1] * len(target_shape)
            view_shape[-(idx + 1)] = -1
            embeddings.append(emb.view(*view_shape).repeat(*target_shape))

        pos_embeddings = torch.cat(embeddings, dim=-1)
        pos_embeddings = pos_embeddings.permute(-1, *range(pos_embeddings.ndim - 1))

        return pos_embeddings
