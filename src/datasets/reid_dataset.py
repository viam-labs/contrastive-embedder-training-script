import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import json

from datasets.stratified_pairs_split import StratifiedPairsSplit

def _validate_filepath(filepath): 
    if not os.path.exists(filepath): 
        new_filepath = os.path.expanduser(filepath)
        if not os.path.exists(new_filepath):
            raise FileNotFoundError(f"File not found {filepath}")
        else:
            return new_filepath
        
    return filepath
    

class REIDPairDataset(Dataset):
    """Re-ID dataset for pair-based contrastive learning."""

    def __init__(self, data: List[Tuple], transform=None, json_path=None):
        """
        Initialize with pre-split data.
        
        Args:
            data: List of (img_1_path, img_2_path, label) tuples
            transform: torchvision transforms to apply
            json_path: Path to JSON file (for resolving relative image paths)
        """
        self.data = data
        self.transform = transform
        self.json_path = json_path 

        # Resolve all paths once during initialization
        self.data = self._resolve_paths(data)
        print(f"Initialized dataset with {len(self.data)} pairs")

    def _get_person_id_from_path(self, image_path):
        """Extract person ID from image file path."""
        try:
            filename = os.path.basename(image_path)
            person_id = filename.split('_')[-1].split('.')[0]
            return person_id
        except Exception as e:
            print(f"Could not process path: {image_path}. Error: {e}")
            return None
        
        
    def _resolve_paths(self, data: List[Tuple]) -> List[Tuple]:
        """Resolve relative paths to absolute paths once during init."""
        if not self.json_path:
            return data
        
        # Get parent directory 
        json_dir = os.path.dirname(self.json_path) 
        
        resolved_data = []
        
        for img1_path, img2_path, label in data:
            # Resolve img1_path
            if not os.path.isabs(img1_path):
                img1_path = os.path.join(json_dir, img1_path)
            
            # Resolve img2_path  
            if not os.path.isabs(img2_path):
                img2_path = os.path.join(json_dir, img2_path)
                
            resolved_data.append((img1_path, img2_path, label))
        
        return resolved_data
    
                
    def _load_image(self, image_path: str):
        """Load image with error handling - no external dependencies."""
        try:
            # Try to load normally
            image = Image.open(image_path).convert('RGB')
            # Force load to catch corruption early
            image.load()
            return image
            
        except (OSError, IOError, Image.UnidentifiedImageError, Image.DecompressionBombError) as e:
            print(f"PIL error for {os.path.basename(image_path)}: {e}")
            
        except Exception as e:
            print(f"Unexpected error for {os.path.basename(image_path)}: {e}")
            return Image.new('RGB', (224, 224), color=(128, 128, 128))
        
            
    def __len__(self) -> int:
        """Return number of pairs in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a pair of images and their label.
        
        Args:
            idx: Index into the dataset
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: (img1, img2, label)
        """
        img1_path, img2_path, label = self.data[idx]
        
        # Load images
        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)
        
        # Apply transforms if provided
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return img1, img2, torch.tensor(label, dtype=torch.float32)


def get_reid_datasets(cfg):
    """
    Create train, validation, and test datasets for Re-ID pair dataset.
    
    Config file is your dataset.yaml in configs/dataset
    
    Args:
        cfg: Configuration object containing dataset parameters
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: (train_dataloader, val_dataloader, test_dataloader)
    """
    
    #Confirm json file exists 
    json_path = _validate_filepath(cfg.source_json_path)
    
    # Load pairs from JSON once
    with open(json_path, "r") as f:
        pairs_data = json.load(f)
    
    # Get all splits at once
    train_data, val_data, test_data = StratifiedPairsSplit.get_splits(
        pairs_data=pairs_data,
        train_size=cfg.train_size,
        val_size=cfg.val_size, 
        test_size=cfg.test_size,
        random_state=cfg.random_state
    ) 
    
    # Define transforms with optional augmentation
    if hasattr(cfg, 'augmentation') and hasattr(cfg.augmentation, 'train'):
        # PIL Image transforms
        train_transform = transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.RandomHorizontalFlip(p=cfg.augmentation.train.horizontal_flip),
            transforms.ColorJitter(
                brightness=cfg.augmentation.train.color_jitter.brightness,
                contrast=cfg.augmentation.train.color_jitter.contrast,
                saturation=cfg.augmentation.train.color_jitter.saturation,
                hue=cfg.augmentation.train.color_jitter.hue
            ),
            # Convert to tensor
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std),
            # Tensor transforms (important order, as RandomErasing works on tensors)
            transforms.RandomErasing(
                p=cfg.augmentation.train.random_erasing.probability,
                scale=tuple(cfg.augmentation.train.random_erasing.scale),
                ratio=tuple(cfg.augmentation.train.random_erasing.ratio)
            )
        ])
    else:
        # Use minimal augmentation when none specified
        train_transform = transforms.Compose([
            transforms.Resize(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=cfg.mean, std=cfg.std)
        ])
        
    eval_transform = transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg.mean, std=cfg.std)
    ])
    
    # Create datasets with pre-split data 
    train_dataset = REIDPairDataset(train_data, transform=train_transform, json_path=json_path)
    val_dataset = REIDPairDataset(val_data, transform=eval_transform, json_path=json_path)
    test_dataset = REIDPairDataset(test_data, transform=eval_transform, json_path=json_path)
        
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        use_pin_memory = False  # pin_memory can cause issues on MPS
    else:
        use_pin_memory = cfg.pin_memory and torch.cuda.is_available()
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle.train,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory, 
        drop_last=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle.val,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle.test,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory
    )
    
    # Print statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Train: {len(train_data)} pairs")
    print(f"Val: {len(val_data)} pairs") 
    print(f"Test: {len(test_data)} pairs")
    
    stats = StratifiedPairsSplit.get_split_stats()
    print(f"Person ID splits: {stats}")
    
    return train_dataloader, val_dataloader, test_dataloader
