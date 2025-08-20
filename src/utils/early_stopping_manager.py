#!/usr/bin/env python3

import logging
import numpy as np

logger = logging.getLogger(__name__)

class EarlyStoppingManager:
    """Handles early stopping state management."""
    
    @staticmethod
    def load_early_stopping_state(checkpoint):
        """Load early stopping state from checkpoint."""
        best_score = None
        epochs_without_improvement = None
        
        if 'best_score' in checkpoint:
            try:
                best_score = checkpoint['best_score']
                logger.info(f"Best score loaded: {best_score:.4f}")
            except Exception as e:
                logger.warning(f"Failed to load best score: {e}")
        
        if 'epochs_without_improvement' in checkpoint:
            try:
                epochs_without_improvement = checkpoint['epochs_without_improvement']
                logger.info(f"Early stopping counter loaded: {epochs_without_improvement}")
            except Exception as e:
                logger.warning(f"Failed to load early stopping counter: {e}")
        
        return best_score, epochs_without_improvement
    
    @staticmethod
    def check_early_stopping(val_metrics, monitor_metric, mode, best_score, 
                           min_delta, epochs_without_improvement, patience):
        """Check if early stopping should be triggered."""
        
        # Handle missing monitor metric gracefully
        if monitor_metric not in val_metrics:
            available_metrics = list(val_metrics.keys())
            logger.warning(f"Monitor metric '{monitor_metric}' not found in validation metrics. "
                         f"Available metrics: {available_metrics}")
            
            # Fallback hierarchy
            fallback_metrics = ['auc', 'f1', 'accuracy', 'val_loss']
            fallback_metric = None
            
            for metric in fallback_metrics:
                if metric in val_metrics:
                    fallback_metric = metric
                    break
            
            if fallback_metric is None:
                # Use the first available metric
                fallback_metric = available_metrics[0] if available_metrics else None
            
            if fallback_metric is None:
                logger.error("No metrics available for early stopping!")
                return False, False, best_score, epochs_without_improvement + 1
            
            logger.warning(f"Using fallback metric '{fallback_metric}' for early stopping.")
            monitor_metric = fallback_metric
        
        current_score = val_metrics[monitor_metric]
        is_best = False
        
        # Handle NaN values
        if np.isnan(current_score):
            logger.warning(f"Monitor metric '{monitor_metric}' returned NaN. Skipping early stopping check.")
            return False, False, best_score, epochs_without_improvement + 1
        
        # Initialize best_score if this is the first validation
        if best_score is None or np.isnan(best_score) or np.isinf(best_score):
            best_score = current_score
            epochs_without_improvement = 0
            is_best = True
            logger.info(f"Initialized best score ({monitor_metric}: {best_score:.4f})")
            return is_best, False, best_score, epochs_without_improvement
        
        # Check for improvement
        if mode == 'max':
            if current_score > best_score + min_delta:
                best_score = current_score
                epochs_without_improvement = 0
                is_best = True
            else:
                epochs_without_improvement += 1
        else:  # mode is 'min'
            if current_score < best_score - min_delta:
                best_score = current_score
                epochs_without_improvement = 0
                is_best = True
            else:
                epochs_without_improvement += 1

        if is_best:
            logger.info(f"New best score ({monitor_metric}: {best_score:.4f}) found. Resetting early stopping counter.")
        else:
            logger.debug(f"No improvement. Current: {current_score:.4f}, Best: {best_score:.4f}, "
                        f"Counter: {epochs_without_improvement}/{patience}")
        
        should_stop = epochs_without_improvement >= patience
        
        if should_stop:
            logger.info(f"Early stopping triggered! No improvement in '{monitor_metric}' "
                       f"for {patience} epochs.")
        
        return is_best, should_stop, best_score, epochs_without_improvement