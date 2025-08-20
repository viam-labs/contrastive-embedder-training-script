#!/usr/bin/env python3

import logging

logger = logging.getLogger(__name__)

class TrainingHistoryManager:
    """Handles training history migration and validation."""
    
    @staticmethod
    def migrate_training_history(old_history, eval_interval):
        """Migrate old training history format to include val_epochs tracking."""
        if 'val_epochs' not in old_history:
            old_history['val_epochs'] = []
            for i, epoch in enumerate(old_history.get('epoch', [])):
                if (epoch + 1) % eval_interval == 0 and i < len(old_history.get('val_loss', [])):
                    old_history['val_epochs'].append(epoch)
        
        # Ensure validation arrays have consistent length
        val_len = len(old_history['val_epochs'])
        for key in ['val_loss', 'val_auc', 'val_accuracy']:
            if key in old_history and len(old_history[key]) > val_len:
                old_history[key] = old_history[key][:val_len]
                logger.debug(f"Trimmed {key} to match val_epochs length")
        
        return old_history
    
    @staticmethod
    def load_training_history(checkpoint, eval_interval):
        """Load and migrate training history from checkpoint."""
        if 'training_history' not in checkpoint:
            logger.warning("No training history found in checkpoint")
            return None
        
        try:
            old_history = checkpoint['training_history']
            migrated_history = TrainingHistoryManager.migrate_training_history(old_history, eval_interval)
            
            epochs_count = len(migrated_history['epoch'])
            val_count = len(migrated_history['val_epochs'])
            logger.info(f"Training history loaded: {epochs_count} epochs, {val_count} validation points")
            return migrated_history
        except Exception as e:
            logger.warning(f"Failed to load training history: {e}")
            return None