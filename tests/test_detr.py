import torch
from torch._C import dtype
from detr.model import Detr
from detr.data.dummy_dataset import DummyDetectionDataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def _collate_fn(batch):
    return tuple(zip(*batch))


def test_detection_forward():
    model = Detr(10)
    image = torch.randn(1, 3, 224, 224)  # Image-net shape
    out = model(image)
    return 1


def test_detr_detection_train(tmpdir):
    model = Detr(num_classes=91)

    train_dl = DataLoader(
        DummyDetectionDataset(
            img_shape=(3, 224, 224), num_boxes=1, num_classes=91, num_samples=10
        ),
        collate_fn=_collate_fn,
    )

    valid_dl = DataLoader(
        DummyDetectionDataset(
            img_shape=(3, 224, 224), num_boxes=1, num_classes=91, num_samples=10
        ),
        collate_fn=_collate_fn,
    )

    trainer = pl.Trainer(fast_dev_run=True, default_root_dir=tmpdir)
    trainer.fit(model, train_dl, valid_dl)
    return 1


# def test_segmentation_forward():
#     model = Detr(10, segmentation=True)
#     image = torch.randn(10, 3, 224, 224)  # Image-net shape
#     out = model(image)
#     return 1


def test_detection_step():
    model = Detr(10)

    batch = {
        "data": torch.randn(1, 3, 224, 224),  # Image-net shape
        "padding_mask": torch.zeros(1, 224, 224, dtype=torch.long),
        "label": [
            {
                "boxes": torch.tensor([1, 5, 78, 80]),
                "labels": torch.tensor([5]),
            }
        ],
    }

    model.training_step(batch, 1)
    return 1


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
