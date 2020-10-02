from detr.data.torchvision_datasets import *
from detr.data.coco import *
from detr.data.dummy_dataset import *
from detr.utils.image_ops import pad_to_max_size as _pad
from torch.utils.data.dataloader import default_collate as _default_collate


class DetrDictCollate:
    def __init__(self, key: str = 'data'):
        self.key = key
        
    def __call__(self, data):
        
        # already padded
        if 'padding_mask' in data[0]:
            return _default_collate(data)
        
        collate_separate = [_data.popitem(self.key) for _data in data]
        
        # collate everything else
        collated = _default_collate(data)
        
        tensor, mask = _pad(collate_separate)
        collated[self.key] = tensor
        collated['padding_mask'] = mask
        
        return collated
