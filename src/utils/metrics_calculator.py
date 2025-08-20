#!/usr/bin/env python3

import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix
import mlflow

class MetricsCalculator:
    """Handles comprehensive metrics calculation."""
    
    @staticmethod
    def compute_comprehensive_metrics(distances, labels, epoch, metric_prefix):
        """Computes comprehensive validation metrics."""
        
        # DEBUG: Print initial shapes
        print(f"Input distances shape: {distances.shape}")
        print(f"Input labels shape: {labels.shape}")
        
        # Ensure distances and labels are 1D numpy arrays
        if len(distances.shape) > 1:
            distances = distances.flatten()
            print(f"Flattened distances shape: {distances.shape}")
            
        if len(labels.shape) > 1:
            labels = labels.flatten()
            print(f"Flattened labels shape: {labels.shape}")
        
        # Convert to numpy arrays if they aren't already
        distances = np.asarray(distances)
        labels = np.asarray(labels)
        
        # Ensure they have the same length
        if len(distances) != len(labels):
            min_len = min(len(distances), len(labels))
            distances = distances[:min_len]
            labels = labels[:min_len]
            print(f"Truncated to matching length: {min_len}")
        
        # Final shape verification
        print(f"Final distances shape: {distances.shape}")
        print(f"Final labels shape: {labels.shape}")
        
        # Proceed with metrics calculation
        scores = -distances
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = -roc_thresholds[optimal_idx]
        optimal_tpr, optimal_fpr = tpr[optimal_idx], fpr[optimal_idx]
        
        predictions = (distances <= optimal_threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = fpr[eer_idx]

        # Log metrics with dynamic prefix
        if epoch >= 0:  # Don't log for final evaluation (epoch = -1)
            mlflow.log_metric(f"{metric_prefix}_auc", roc_auc, step=epoch)
            mlflow.log_metric(f"{metric_prefix}_pr_auc", pr_auc, step=epoch)
            mlflow.log_metric(f"{metric_prefix}_precision", precision_score, step=epoch)
            mlflow.log_metric(f"{metric_prefix}_recall", recall_score, step=epoch)
            mlflow.log_metric(f"{metric_prefix}_f1", f1_score, step=epoch)
            mlflow.log_metric(f"{metric_prefix}_eer", eer, step=epoch)
            mlflow.log_metric(f"{metric_prefix}_optimal_tpr", optimal_tpr, step=epoch)
            mlflow.log_metric(f"{metric_prefix}_optimal_fpr", optimal_fpr, step=epoch)
        
        return {
            'auc': roc_auc, 'pr_auc': pr_auc, 'accuracy': accuracy,
            'precision': precision_score, 'recall': recall_score, 'f1': f1_score,
            'eer': eer, 'optimal_threshold': optimal_threshold
        }