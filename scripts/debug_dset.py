from detr.data.torchvision_datasets.coco import CocoDetection
from pathlib import Path


if __name__ == '__main__':
    root = Path('~/Downloads/COCO').expanduser()
    image_set = 'val'
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, transforms=None, return_masks=True)
    print(dataset[0])
    print(len(dataset))
    print('')