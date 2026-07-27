from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from collections import Counter

from PIL import Image

from torch.utils.data import Dataset, DataLoader

from torchvision import transforms

from config import (
    IMAGES_DIR,
    META_DIR,
    SELECTED_CLASSES,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEVICE
)

from data.histogram_equalization import HistogramEqualization

pin_memory = DEVICE == "cuda"

# TRANSFORMS

train_transform = transforms.Compose([
    HistogramEqualization(),

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

test_transform = transforms.Compose([
    HistogramEqualization(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])

class Food101SubsetDataset(Dataset):
    def __init__(self, split, transform=None):
        super().__init__()
        
        self.transform = transform

        self.class_names = SELECTED_CLASSES

        self.class_to_idx = {
            cls: idx
            for idx, cls in enumerate(self.class_names)
        }

        self.samples = []

        if split == "train":
            split_file = META_DIR / "train.txt"

        elif split == "test":
            split_file = META_DIR / "test.txt"
        else:
            raise ValueError("split must be 'train' or 'test'")

        with open(split_file) as f:
            lines = f.read().splitlines()

        for line in lines:
            class_name = line.split("/")[0]

            if class_name not in self.class_names:
                continue
            image_path = IMAGES_DIR / f"{line}.jpg"

            label = self.class_to_idx[class_name]

            self.samples.append(
                (image_path, label)
            )

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_path, label = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# DATASETS
train_dataset = Food101SubsetDataset(
        split="train",
        transform=train_transform
    )

test_dataset = Food101SubsetDataset(
    split="test",
    transform=test_transform
) 

class_names = train_dataset.class_names

# DATALOADERS
train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=pin_memory
)

if __name__ == "__main__":
    print("Train Images:", len(train_dataset))
    print("Test Images:", len(test_dataset))

    print(train_dataset.samples[:5])

    print("class names:", class_names)
    print(len(train_dataset.class_names))

    images, labels = next(iter(train_dataloader))

    print(images.shape)
    print(labels.shape)