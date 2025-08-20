#!/usr/bin/env python3

import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class ProgressiveUnfreezingManager:
    """
    Manages progressive unfreezing strategy during transfer learning.
    
    Handles staged unfreezing of model layers with different learning rates
    to provide more stable training when transferring from larger to smaller models.
    """
    
    def __init__(self, model, cfg):
        """
        Initialize progressive unfreezing manager.
        
        Args:
            model: PyTorch model to manage
            cfg: Configuration object containing progressive unfreezing settings
        """
        self.model = model
        self.cfg = cfg
        self.current_stage = 0
        self.stage_start_epoch = 0
        self.stages = self._get_stages()
        
        if self.stages:
            logger.info(f"Progressive unfreezing enabled with {len(self.stages)} stages")
            self._log_stage_info()
        else:
            logger.info("Progressive unfreezing disabled")
    
    def _get_stages(self) -> List:
        """Extract progressive unfreezing stages from config."""
        if (hasattr(self.cfg, 'transfer_learning') and 
            hasattr(self.cfg.transfer_learning, 'progressive_unfreezing') and
            self.cfg.transfer_learning.progressive_unfreezing.enabled):
            return self.cfg.transfer_learning.progressive_unfreezing.stages
        return []
    
    def _log_stage_info(self):
        """Log information about all stages."""
        for i, stage in enumerate(self.stages):
            logger.info(f"Stage {i+1}: {stage.epochs} epochs, "
                       f"freeze {stage.freeze_layers} layer groups, lr={stage.lr}")
    
    def should_update_stage(self, current_epoch: int) -> bool:
        """
        Check if we should advance to the next training stage.
        
        Args:
            current_epoch: Current training epoch
            
        Returns:
            True if stage should be updated, False otherwise
        """
        if not self.stages or self.current_stage >= len(self.stages):
            return False
            
        epochs_in_current_stage = current_epoch - self.stage_start_epoch
        return epochs_in_current_stage >= self.stages[self.current_stage].epochs
    
    def update_stage(self, current_epoch: int, optimizer) -> bool:
        """
        Update to next training stage if conditions are met.
        
        Args:
            current_epoch: Current training epoch
            optimizer: PyTorch optimizer to update learning rates
            
        Returns:
            True if stage was updated, False otherwise
        """
        if not self.should_update_stage(current_epoch):
            return False
            
        self.current_stage += 1
        self.stage_start_epoch = current_epoch
        
        if self.current_stage < len(self.stages):
            stage = self.stages[self.current_stage]
            
            # Update frozen layers
            frozen_count = self._freeze_layers(stage.freeze_layers)
            
            # Update learning rates
            self._update_learning_rates(optimizer, stage.lr)
            
            logger.info(f"Advanced to stage {self.current_stage + 1}/{len(self.stages)}: "
                       f"froze {frozen_count} layer groups, lr={stage.lr}")
            return True
        
        return False
    
    def _freeze_layers(self, num_frozen_groups: int) -> int:
        """
        Freeze specified number of layer groups.
        
        Args:
            num_frozen_groups: Number of layer groups to freeze from the beginning
            
        Returns:
            Number of parameters that were frozen
        """
        layer_groups = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
        frozen_groups = layer_groups[:num_frozen_groups]
        
        frozen_params = 0
        total_params = 0
        
        for name, param in self.model.named_parameters():
            total_params += 1
            if any(group in name for group in frozen_groups):
                param.requires_grad = False
                frozen_params += 1
            else:
                param.requires_grad = True
        
        logger.debug(f"Frozen {frozen_params}/{total_params} parameters from groups: {frozen_groups}")
        return len(frozen_groups)
    
    def _update_learning_rates(self, optimizer, new_lr: float):
        """
        Update optimizer learning rates for current stage.
        
        Args:
            optimizer: PyTorch optimizer
            new_lr: New base learning rate
        """
        # Handle differential learning rates if configured
        if len(optimizer.param_groups) > 1:
            # Assume first group is backbone, second is head
            optimizer.param_groups[0]['lr'] = new_lr * 0.1  # Backbone gets 10x smaller LR
            optimizer.param_groups[1]['lr'] = new_lr        # Head gets full LR
            logger.debug(f"Updated LRs - Backbone: {new_lr * 0.1:.2e}, Head: {new_lr:.2e}")
        else:
            # Single parameter group
            optimizer.param_groups[0]['lr'] = new_lr
            logger.debug(f"Updated LR: {new_lr:.2e}")
    
    def get_current_stage_info(self) -> Dict:
        """
        Get information about current training stage.
        
        Returns:
            Dictionary with current stage information
        """
        if not self.stages or self.current_stage >= len(self.stages):
            return {
                'stage': 'none',
                'stage_number': 0,
                'total_stages': len(self.stages),
                'frozen_layers': 0,
                'lr': None
            }
        
        current = self.stages[self.current_stage]
        return {
            'stage': 'active',
            'stage_number': self.current_stage + 1,
            'total_stages': len(self.stages),
            'frozen_layers': current.freeze_layers,
            'lr': current.lr
        }
    
    def is_enabled(self) -> bool:
        """Check if progressive unfreezing is enabled."""
        return len(self.stages) > 0