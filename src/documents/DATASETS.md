Datasets
This template provides a flexible framework for training on pair-based datasets for contrastive learning. Here's how to integrate a new dataset into the training pipeline.

Supported Data Format
The template supports pair-based contrastive learning with the following format:

Direct Pairs: [(x_i, x_j, label_int)]

x_i, x_j: Image paths or PIL Images

label_int: 1 for same person, 0 for different person

How to Create a New Dataset
Step 1: Create the Dataset Class
Copy the provided template and create your own dataset class, inheriting from GenericPairDataset. Implement the **init**, **len**, and **getitem** methods to load your data.

Bash

cp src/datasets/generic_dataset.py src/datasets/my_dataset.py
Python

# src/datasets/my_dataset.py

from src.datasets.generic_dataset import GenericPairDataset
from torchvision import transforms
from PIL import Image
import torch

class MyPairDataset(GenericPairDataset):
def **init**(self, cfg, transform=None):
super().**init**(cfg, transform) # Load your dataset data here
self.data = [
("path/to/img1.jpg", "path/to/img2.jpg", 1), # same person
("path/to/img3.jpg", "path/to/img4.jpg", 0), # different person
# ... more pairs
]

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.data[idx]

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

def get_my_datasets(cfg):
transform = transforms.Compose([
transforms.Resize((224, 224)),
transforms.ToTensor(),
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    train_dataset = MyPairDataset(cfg, transform=transform)

    # Create dataloaders for train, validation, and test sets...
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    # ... create validation and test loaders similarly

    return train_dataloader, val_dataloader, test_dataloader

Step 2: Create the Configuration File
Create a new YAML configuration file for your dataset.

Code snippet

name: my_dataset
batch_size: 32
num_workers: 4
pin_memory: true
num_pairs_per_epoch: 10000
Step 3: Update the Training Script
Update the \_get_datasets method in src/train.py to include your new dataset function.

Python

# In EnhancedReIDTrainer.\_get_datasets

def \_get_datasets(self):
dataset_map = {
"mnist_pairs": get_mnist_datasets,
"generic_pairs": get_generic_datasets,
"reid_pairs": get_reid_datasets,
"my_dataset": get_my_datasets # Add your new dataset function here
}
dataset_fn = dataset_map.get(self.cfg.dataset.name)
if not dataset_fn:
logger.error(f"Unknown dataset: {self.cfg.dataset.name}")
raise ValueError(f"Unknown dataset: {self.cfg.dataset.name}")
return dataset_fn(self.cfg.dataset)
Step 4: Run with Your Dataset
Use the command-line override to select your new dataset.

Bash

python src/train.py dataset=my_dataset
