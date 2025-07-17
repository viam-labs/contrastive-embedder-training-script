from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class GenericPairDataset(Dataset):
    """
    Generic template for pair-based contrastive learning datasets.

    This template supports the data format: [(x_i, x_j, label_int)]
    where x_i and x_j are two pictures (cropped images of people),
    and label is 1 for a positive pair and 0 for a negative pair.

    To use this template, inherit from this class and implement:
    - __getitem__(self, idx): Return (img1, img2, label) for a given index
    - __len__(self): Return the total number of pairs in the dataset

    The _load_dataset() and _load_image() methods are provided as examples
    but are not required - you can implement data loading however you prefer.
    """

    def __init__(self):
        """
        Initialize the generic pair dataset.
        """

        # TODO: USER MUST IMPLEMENT - Load your dataset here
        self.data = self._load_dataset()

    def _load_dataset(self) -> List:
        """
        OPTIONAL: Example method for loading dataset data.

        This method is provided as an example of how you might load your dataset.
        You can override this method, or implement data loading directly in __getitem__.

        Returns:
            List of tuples [(x_i, x_j, label_int), ...]
            Where x_i, x_j are image paths or PIL Images, label_int is 0 or 1

        Example:
            return [
                ("path/to/img1.jpg", "path/to/img2.jpg", 1),  # same person
                ("path/to/img3.jpg", "path/to/img4.jpg", 0),  # different person
                ...
            ]
        """
        raise NotImplementedError(
            "This is an example method - implement data loading as needed"
        )

    def _load_image(self, image_path: str):
        """
        OPTIONAL: Example method for loading images from file paths.

        This method is provided as an example of how you might load images.
        You can override this method, or implement image loading directly in __getitem__.

        Args:
            image_path (str): Path to the image file

        Returns:
            Image in the format expected by your transforms (PIL Image, numpy array, etc.)
        """
        # Example implementation:
        # from PIL import Image
        # return Image.open(image_path).convert('RGB')
        raise NotImplementedError(
            "This is an example method - implement image loading as needed"
        )

    def __len__(self) -> int:
        """Return number of pairs in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of images and their label.

        Args:
            idx (int): Index (not used, pairs are generated randomly)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (img1, img2, label)
        """
        # Randomly select a pair from the dataset
        pair_idx = np.random.randint(0, len(self.data))
        img1_path, img2_path, label = self.data[pair_idx]

        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_generic_datasets(cfg):
    """
    TODO: USER MUST IMPLEMENT

    Create train and validation datasets for the generic pair dataset.

    Args:
        cfg: Configuration object containing dataset parameters

    Returns:
        Tuple[DataLoader, DataLoader]: (train_dataloader, val_dataloader)
    """
    # TODO: USER MUST IMPLEMENT - Define your transforms here
    # Example:
    # from torchvision import transforms
    # transform = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    transform = None  # Define your transforms here

    # TODO: USER MUST IMPLEMENT - Create train dataset
    train_dataset = GenericPairDataset()

    # TODO: USER MUST IMPLEMENT - Create validation dataset
    val_dataset = GenericPairDataset()

    # TODO: USER MUST IMPLEMENT - Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )

    return train_dataloader, val_dataloader
