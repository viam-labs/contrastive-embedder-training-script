#!/usr/bin/env python3

import torch
import mlflow
import glob
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Handles all checkpoint-related operations."""
    
    def __init__(self, save_dir, device):
        self.save_dir = Path(save_dir)
        self.device = device
    
    def find_latest_checkpoint(self):
        """Find the latest checkpoint in the save directory."""
        checkpoint_pattern = str(self.save_dir / "checkpoint_epoch_*.pth.tar")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if not checkpoints:
            return None
            
        epoch_numbers = []
        for checkpoint in checkpoints:
            try:
                epoch_num = int(checkpoint.split('_epoch_')[1].split('.')[0])
                epoch_numbers.append((epoch_num, checkpoint))
            except (IndexError, ValueError):
                continue
                
        if epoch_numbers:
            latest_epoch, latest_checkpoint = max(epoch_numbers, key=lambda x: x[0])
            return latest_checkpoint
        return None
    
    def load_model_state(self, model, checkpoint):
        """Load model state from checkpoint."""
        try:
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Model state loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model state: {e}")
            return False
    
    def load_optimizer_state(self, optimizer, checkpoint):
        """Load optimizer state from checkpoint."""
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("Optimizer state loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load optimizer state: {e}")
            return False
    
    def load_scheduler_state(self, scheduler, checkpoint):
        """Load scheduler state from checkpoint."""
        if not scheduler or 'scheduler' not in checkpoint:
            return False
            
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("Scheduler state loaded successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")
            return False
    
    def create_checkpoint_data(self, model, optimizer, scheduler, epoch, train_loss, 
                             val_loss, val_metrics, cfg, training_history, best_score, 
                             epochs_without_improvement):
        """Create checkpoint data dictionary."""
        checkpoint_data = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': cfg,
            'training_history': training_history,
            'best_score': best_score,
            'epochs_without_improvement': epochs_without_improvement
        }
        
        if scheduler:
            checkpoint_data['scheduler'] = scheduler.state_dict()
        
        return checkpoint_data
    
    def save_checkpoint_file(self, checkpoint_data, filepath):
        """Save checkpoint data to file and log to MLflow."""
        try:
            torch.save(checkpoint_data, filepath)
            mlflow.log_artifact(str(filepath))
            logger.info(f"Checkpoint saved: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint {filepath}: {e}")
            return False