# models/__init__.py

"""
Models module for the contrastive embedding training framework.
This module provides access to various neural network architectures for person re-identification.
"""

from .simple_cnn import SimpleCNN
from .lightweight_embedder import LightweightEmbedder

# OSNet model variants - import from osnet_ain.py
from .osnet_ain import (
    osnet_ain_x0_25,
    osnet_ain_x0_5, 
    osnet_ain_x0_75,
    osnet_ain_x1_0
)

__all__ = [
    'SimpleCNN',
    'LightweightEmbedder', 
    'osnet_ain_x0_25',
    'osnet_ain_x0_5',
    'osnet_ain_x0_75', 
    'osnet_ain_x1_0'
]