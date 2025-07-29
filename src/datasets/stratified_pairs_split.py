import json
import os
from typing import List, Set

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split

def get_person_id_from_path(image_path: str) -> str:
    """Extract person ID from image file path."""
    try:
        return os.path.splitext(os.path.basename(image_path))[0].split('_')[-1]
    except Exception:
        return None

class StratifiedPairsSplit:
    """Handles stratified splitting of Re-ID dataset by person ID with caching."""
    
    _processed_data = None
    _id_splits = {}

    @classmethod
    def get_splits(cls, pairs_data: List, train_size=0.65, val_size=0.15, test_size=0.2, random_state=42):
        """Get all three splits as (img_1, img_2, label) tuples."""
        if not cls._processed_data:
            cls._process_and_split(pairs_data, train_size, val_size, test_size, random_state)

        splits = {'train': [], 'val': [], 'test': []}
        
        for p in cls._processed_data:
            # Extract image paths and original label
            original_pair = p['original_pair']
            if len(original_pair) == 3:
                img_1, img_2, label = original_pair
            elif len(original_pair) == 2:
                img_1, img_2 = original_pair
                label = 1 if p['id1'] == p['id2'] else 0  # Fallback to generated label
            else:
                continue  # Skip invalid pairs
            
            # Determine which split this pair belongs to
            if p['id1'] in cls._id_splits['train'] and p['id2'] in cls._id_splits['train']:
                splits['train'].append((img_1, img_2, label))
            elif p['id1'] in cls._id_splits['val'] and p['id2'] in cls._id_splits['val']:
                splits['val'].append((img_1, img_2, label))
            elif p['id1'] in cls._id_splits['test'] and p['id2'] in cls._id_splits['test']:
                splits['test'].append((img_1, img_2, label))
        
        return splits['train'], splits['val'], splits['test']
    
    
    @classmethod
    def _process_and_split(cls, pairs_data: List, train_size: float, val_size: float, test_size: float, random_state: int):
        """Process pairs data and perform stratified split by person ID."""
        cls._processed_data = []
        all_person_ids = set()
        
        for pair in pairs_data:
            id1 = get_person_id_from_path(pair[0])
            id2 = get_person_id_from_path(pair[1])
            label = pair[2]
            
            if id1 and id2:
                cls._processed_data.append({'id1': id1, 'id2': id2, 'original_pair': pair})
                all_person_ids.add(id1)
                all_person_ids.add(id2)
        
        cls._split_ids(list(all_person_ids), train_size, val_size, test_size, random_state)

    @classmethod
    def _split_ids(cls, all_ids: List[str], train_size: float, val_size: float, test_size: float, random_state: int):
        """Split person IDs maintaining overall dataset percentages with validation."""
        total_size = train_size + val_size + test_size
        if not (0.999 <= total_size <= 1.001):
            raise ValueError(f"Split sizes must sum to 1.0, got {total_size}")

        train_val_ids, test_ids = train_test_split(
            all_ids, test_size=test_size, random_state=random_state
        )

        val_size_relative = val_size / (train_size + val_size)
        
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=val_size_relative, random_state=random_state
        )

        cls._id_splits = {
            'train': set(train_ids),
            'val': set(val_ids),
            'test': set(test_ids)
        }
        
        # Validate no person ID overlap between splits
        cls._validate_split_isolation()

    @classmethod
    def _validate_split_isolation(cls):
        """Validate that person IDs are completely isolated between splits."""
        train_ids = cls._id_splits['train']
        val_ids = cls._id_splits['val']
        test_ids = cls._id_splits['test']
        
        assert len(train_ids.intersection(val_ids)) == 0, "Error: Train and Validation IDs overlap!"
        assert len(train_ids.intersection(test_ids)) == 0, "Error: Train and Test IDs overlap!"
        assert len(val_ids.intersection(test_ids)) == 0, "Error: Validation and Test IDs overlap!"


    @classmethod
    def reset_splits(cls):
        """Clear cached splits."""
        cls._processed_data = None
        cls._id_splits = {} 

    @classmethod
    def get_split_stats(cls):
        """Return split statistics."""
        if not cls._id_splits:
            return "No splits available. Run get_splits() first."
        return {split_name: len(ids) for split_name, ids in cls._id_splits.items()}