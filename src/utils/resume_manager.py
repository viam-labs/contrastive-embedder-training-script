#!/usr/bin/env python3

import torch
import logging
from pathlib import Path
from .checkpoint_manager import CheckpointManager
from .training_history_manager import TrainingHistoryManager
from .early_stopping_manager import EarlyStoppingManager

logger = logging.getLogger(__name__)

class TransferLearningManager:
    """Handles transfer learning specific operations."""
    
    @staticmethod
    def load_transfer_learning_info(checkpoint):
        """Load and log transfer learning information from checkpoint."""
        if 'transfer_learning_info' not in checkpoint:
            return False
        
        try:
            tl_info = checkpoint['transfer_learning_info']
            logger.info("Transfer learning checkpoint detected")
            
            source = tl_info.get('source_variant', 'unknown')
            target = tl_info.get('target_variant', 'unknown') 
            method = tl_info.get('method', 'unknown')
            
            logger.info(f"Transfer details: {source} -> {target}, method: {method}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load transfer learning info: {e}")
            return False

class ResumeManager:
    """Main class that orchestrates the resume process."""
    
    def __init__(self, save_dir, device, eval_interval):
        self.checkpoint_manager = CheckpointManager(save_dir, device)
        self.history_manager = TrainingHistoryManager()
        self.early_stopping_manager = EarlyStoppingManager()
        self.transfer_learning_manager = TransferLearningManager()
        self.eval_interval = eval_interval
    
    def auto_resume(self, model, optimizer, scheduler, cfg):
        """Automatically resume from the latest checkpoint if available."""
        latest_checkpoint = self.checkpoint_manager.find_latest_checkpoint()
        if latest_checkpoint:
            logger.info(f"Auto-resuming from latest checkpoint: {latest_checkpoint}")
            return self.resume_from_checkpoint(latest_checkpoint, model, optimizer, scheduler, cfg)
        else:
            logger.info("No checkpoint found for auto-resume. Starting from scratch.")
            return None
    
    def resume_from_checkpoint(self, checkpoint_path, model, optimizer, scheduler, cfg):
        """Resume training from a checkpoint."""
        if isinstance(checkpoint_path, str):
            checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return self._handle_resume_failure()
        
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.checkpoint_manager.device)
            
            # Load components independently
            model_loaded = self.checkpoint_manager.load_model_state(model, checkpoint)
            optimizer_loaded = self.checkpoint_manager.load_optimizer_state(optimizer, checkpoint)
            scheduler_loaded = self.checkpoint_manager.load_scheduler_state(scheduler, checkpoint)
            
            # Load epoch state
            start_epoch = checkpoint.get('epoch', 0) + 1
            logger.info(f"Starting from epoch: {start_epoch}")
            
            # Load training history
            training_history = self.history_manager.load_training_history(checkpoint, self.eval_interval)
            
            # Load early stopping state
            best_score, epochs_without_improvement = self.early_stopping_manager.load_early_stopping_state(checkpoint)
            
            # Load transfer learning info
            transfer_info_loaded = self.transfer_learning_manager.load_transfer_learning_info(checkpoint)
            
            successful_loads = sum([model_loaded, optimizer_loaded, scheduler_loaded, 
                                  training_history is not None, best_score is not None, 
                                  epochs_without_improvement is not None])
            logger.info(f"Resume complete: {successful_loads}/6 components loaded")
            logger.info(f"Remaining epochs: {cfg.max_epochs - start_epoch}")
            
            return {
                'start_epoch': start_epoch,
                'training_history': training_history,
                'best_score': best_score,
                'epochs_without_improvement': epochs_without_improvement
            }
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return self._handle_resume_failure()
    
    def _handle_resume_failure(self):
        """Handle failed resume attempt."""
        logger.info("Starting training from scratch")
        return None