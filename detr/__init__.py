import torchvision
from detr.data import *
from detr.losses import *
from detr.model import *
from detr.utils import *

# class CocoDetectionDataset(torchvision.datasets.CocoDetection):
#     def __init__(self, root: str, annotation_file: str):
#         super().__init__(root, annFile=annotation_file, transform=torchvision.transforms.ToTensor())

#     def __getitem__(self, index: int):
#         sample = super().__getitem__(index)
