import torch
from torch.utils.data import Dataset, DataLoader

__all__ = ["DummyDetectionDataset"]


class DummyDetectionDataset(Dataset):
    def __init__(
        self, img_shape=(3, 256, 256), num_boxes=1, num_classes=2, num_samples=10000
    ):
        super().__init__()
        self.img_shape = img_shape
        self.num_samples = num_samples
        self.num_boxes = num_boxes
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def _random_bbox(self):
        c, h, w = self.img_shape
        xs = torch.randint(w, (2,))
        ys = torch.randint(h, (2,))
        return [min(xs), min(ys), max(xs), max(ys)]

    def __getitem__(self, idx):
        batch = {}
        img = torch.rand(self.img_shape)
        boxes = torch.tensor(
            [self._random_bbox() for _ in range(self.num_boxes)], dtype=torch.float32)

        labels = torch.randint(self.num_classes, (self.num_boxes,), dtype=torch.long)

        batch["data"] = img
        batch["label"] = [{"boxes": boxes}, {"labels": labels}]
        return batch

# if __name__ == "__main__":
#     ds = DummyDetectionDataset()
#     print(ds[0])

