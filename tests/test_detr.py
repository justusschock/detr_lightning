import torch
from torch._C import dtype
from detr import Detr


def test_detection_forward():
    model = Detr(10)
    image = torch.randn(1, 3, 224, 224)  # Image-net shape
    out = model(image)
    return 1

# def test_segmentation_forward():
#     model = Detr(10, segmentation=True)
#     image = torch.randn(10, 3, 224, 224)  # Image-net shape
#     out = model(image)
#     return 1


# def test_detection_step():
#     model = Detr(10)

#     batch = {
#         "data": torch.randn(1, 3, 224, 224),  # Image-net shape
#         "padding_mask": torch.zeros(1, 224, 224, dtype=torch.long),
#         "label": [
#             {
#                 "boxes": torch.tensor([1, 5, 78, 80]),
#                 "labels": torch.tensor([5]),
#             }
#         ],
#     }

#     model.training_step(batch, 1)
#     return 1


# def test_segmentation_step():
#     model = Detr(10, segmentation=True)

#     batch = {
#         "data": torch.randn(1, 3, 224, 224),  # Image-net shape
#         "padding_mask": torch.zeros(1, 224, 224, dtype=torch.long),
#         "label": [
#             {
#                 "boxes": torch.tensor([[1, 5, 78, 80]]),
#                 "labels": torch.tensor([5]),
#                 "masks": torch.randint(10, (224, 224), dtype=torch.long),
#             },
#         ],
#     }

#     model.training_step(batch, 1)
#     return 1
