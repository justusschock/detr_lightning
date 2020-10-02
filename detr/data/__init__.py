from detr.data.torchvision_datasets import *
from detr.data.coco import *
from detr.data.dummy_dataset import *
from detr.utils.image_ops import pad_to_max_size as _pad
from torch.utils.data.dataloader import default_collate as _default_collate


class DetrDictCollate:
    def __init__(self, key: str = 'data', wrapped_collate: Callable = _default_collate):
        self.key = key
        self.wrapped_collate = wrapped_collate

    def __call__(self, data):

        # already padded
        if 'padding_mask' in data[0]:
            return self._wrapped_collate(data)

        collate_separate = [_data.pop(self.key) for _data in data]

        # collate everything else
        collated = self._wrapped_collate(data)

        tensor, mask = _pad(collate_separate)
        collated[self.key] = tensor
        collated['padding_mask'] = mask

        return collated
