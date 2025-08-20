#!/usr/bin/env python3

# Standard library imports
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Tuple

# Third-party library imports
import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

# Local application/library specific imports
from datasets.generic_dataset import get_generic_datasets
from datasets.mnist_dataset import get_mnist_datasets
from datasets.reid_dataset import get_reid_datasets
from losses.contrastive_loss import ContrastiveLoss
from models import (osnet_ain_x0_25, osnet_ain_x0_5, osnet_ain_x0_75,
                    osnet_ain_x1_0)
from models.lightweight_embedder import LightweightEmbedder
from models.simple_cnn import SimpleCNN
from utils.checkpoint_manager import CheckpointManager
from utils.early_stopping_manager import EarlyStoppingManager
from utils.metrics_calculator import MetricsCalculator
from utils.plotter import Plotter
from utils.progressive_unfreezing_manager import ProgressiveUnfreezingManager
from utils.resume_manager import ResumeManager
from utils.transfer_learning import (add_transfer_info_to_checkpoint,
                                     apply_transfer_learning, clip_gradients,
                                     freeze_layers, create_scheduler,
                                     get_optimizer_with_differential_lr)

# NEW: Import knowledge distillation utilities
from utils.knowledge_distillation import apply_knowledge_distillation, add_distillation_info_to_checkpoint

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedReIDTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if torch.backends.mps.is_available() 
                                 else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize core components
        self.model = self._get_model()
        self.progressive_manager = ProgressiveUnfreezingManager(self.model, cfg)

        self.model = apply_transfer_learning(self.model, cfg, self.device)
        freeze_layers(self.model, cfg)
        
        # Initialize loss function
        base_loss = self._get_loss()
        
        # Apply knowledge distillation if enabled via flag
        self.teacher_model = None
        if self._is_knowledge_distillation_enabled():
            logger.info("Knowledge distillation flag is enabled - setting up distillation")
            self.teacher_model, self.criterion = apply_knowledge_distillation(base_loss, cfg, self.device)
        else:
            logger.info("Knowledge distillation flag is disabled - using standard training")
            self.criterion = base_loss
        
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        self.train_loader, self.val_loader, self.test_loader = self._get_datasets()
                
        # Setup directories
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.save_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Initialize core components
        self._setup_mlflow()
        
        # Initialize training state
        self.training_history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_accuracy': [],
            'learning_rate': [], 'grad_norm': [], 'epoch_time': [], 'val_epochs': []
        }
        
        # Training configuration
        self.eval_interval = self.cfg.get('evaluation_interval', 1) 
        self.checkpoint_interval = self.cfg.get('checkpoint_epoch', 1)
        
        # Early stopping configuration
        self.patience = self.cfg.early_stopping.patience
        self.monitor_metric = self.cfg.early_stopping.monitor
        self.mode = self.cfg.early_stopping.mode
        self.min_delta = self.cfg.early_stopping.min_delta
        
        # Initialize utilities
        self.checkpoint_manager = CheckpointManager(self.save_dir, self.device)
        self.resume_manager = ResumeManager(self.save_dir, self.device, self.eval_interval)
        
        # Initialize training state
        self.start_epoch = 0
        self.best_score = -float('inf') if self.mode == 'max' else float('inf')
        self.epochs_without_improvement = 0
        
        # Handle resume
        self._handle_resume()
    
    def _is_knowledge_distillation_enabled(self) -> bool:
        """
        Check if knowledge distillation is enabled via configuration flag
        
        Returns:
            bool: True if knowledge distillation should be used
        """
        return (
            hasattr(self.cfg, 'knowledge_distillation') and 
            self.cfg.knowledge_distillation.get('enabled', False)
        )
    
    def _handle_resume(self):
        """Handle resume functionality based on configuration."""
        resume_data = None
        
        if hasattr(self.cfg, 'resume_from_checkpoint') and self.cfg.resume_from_checkpoint:
            logger.info("Resume from specific checkpoint requested")
            resume_data = self.resume_manager.resume_from_checkpoint(
                self.cfg.resume_from_checkpoint, self.model, self.optimizer, self.scheduler, self.cfg
            )
        elif hasattr(self.cfg, 'auto_resume') and self.cfg.auto_resume:
            logger.info("Auto-resume enabled")
            resume_data = self.resume_manager.auto_resume(self.model, self.optimizer, self.scheduler, self.cfg)
        
        if resume_data:
            self.start_epoch = resume_data['start_epoch']
            if resume_data['training_history']:
                self.training_history = resume_data['training_history']
            if resume_data['best_score'] is not None:
                self.best_score = resume_data['best_score']
            if resume_data['epochs_without_improvement'] is not None:
                self.epochs_without_improvement = resume_data['epochs_without_improvement']
            
            logger.info(f"Training will resume from epoch {self.start_epoch}")
            logger.info(f"Current best {self.monitor_metric}: {self.best_score:.4f}")
    
    def train(self):
        logger.info("Starting enhanced training with comprehensive monitoring...")
        logger.info(f"Model: {self.cfg.model.name}, Variant: {self.cfg.model.params.variant}")
        logger.info(f"Dataset: {self.cfg.dataset.name}")
        logger.info(f"Learning Rate: {self.cfg.learning_rate}, Max Epochs: {self.cfg.max_epochs}")
        
        # Log knowledge distillation status
        if self._is_knowledge_distillation_enabled():
            logger.info(f"Knowledge Distillation: ENABLED (Teacher: {self.cfg.knowledge_distillation.get('teacher_variant', 'x1_0')})")
        else:
            logger.info("Knowledge Distillation: DISABLED")
        
        logger.info("="*60)
        
        for epoch in range(self.start_epoch, self.cfg.max_epochs):
            train_loss, epoch_time = self.train_epoch(epoch)
            early_stopping = False
            val_loss = None
            val_metrics = None
            is_best = False
            
            if (epoch + 1) % self.eval_interval == 0:
                val_loss, val_metrics = self.validate_epoch(epoch)
                # Add 'val_' prefix to metrics for early stopping compatibility
                val_metrics_prefixed = {}
                for key, value in val_metrics.items():
                    val_metrics_prefixed[f'val_{key}'] = value
                    val_metrics_prefixed[key] = value  # Keep original keys too

                # Use prefixed metrics for monitoring
                val_metrics = val_metrics_prefixed
                                
                self._log_model_stats(epoch)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    elif hasattr(self.scheduler, 'step'):
                        self.scheduler.step()

                
                # Track validation epoch and metrics
                self.training_history['val_epochs'].append(epoch)
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_auc'].append(val_metrics['auc'])
                self.training_history['val_accuracy'].append(val_metrics['accuracy'])
                
                # MLflow logging
                mlflow.log_metric("epoch_val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_auc", val_metrics['auc'], step=epoch)
                mlflow.log_metric("val_accuracy", val_metrics['accuracy'], step=epoch)
                
                # Separate best model tracking (independent of early stopping min_delta)
                current_score = val_metrics[self.monitor_metric]
                is_best_separate = False
                if self.mode == 'max':
                    is_best_separate = current_score > self.best_score
                else:
                    is_best_separate = current_score < self.best_score
                
                # Check early stopping
                is_best, early_stopping, self.best_score, self.epochs_without_improvement = (
                    EarlyStoppingManager.check_early_stopping(
                        val_metrics, self.monitor_metric, self.mode, self.best_score,
                        self.min_delta, self.epochs_without_improvement, self.patience
                    )
                )
                
                # Use the separate best tracking for actual best score updates
                if is_best_separate:
                    self.best_score = current_score
                    is_best = True 
                    
                
                logger.info(f"Epoch {epoch+1}/{self.cfg.max_epochs} - Train Loss: {train_loss:.4f}, "
                           f"Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}, "
                           f"Val F1: {val_metrics['f1']:.4f}, Epoch Time: {epoch_time:.1f}s, "
                           f"LR: {current_lr:.6f}")
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{self.cfg.max_epochs} - Train Loss: {train_loss:.4f}, "
                           f"Epoch Time: {epoch_time:.1f}s, LR: {current_lr:.6f}")
            
            # Track training metrics
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['epoch_time'].append(epoch_time)
            
            # MLflow logging
            mlflow.log_metric("epoch_train_loss", train_loss, step=epoch)
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)
            
            # Save checkpoint
            if (epoch + 1) % self.checkpoint_interval == 0 or early_stopping:
                self.save_checkpoint(epoch, train_loss, val_loss or 0.0, val_metrics or {}, is_best)
            
            # Early stopping check
            if early_stopping:
                logger.info(f"Early stopping triggered! No improvement in '{self.monitor_metric}' "
                           f"for {self.patience} epochs. Stopping training.")
                Plotter.create_training_history_plot(self.training_history, self.plots_dir)
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation score for '{self.monitor_metric}': {self.best_score:.4f}")
        
        if self.cfg.run_eval: 
            self.final_evaluation()
        
        mlflow.end_run()

    def train_epoch(self, epoch):
        """Runs a single training epoch with detailed monitoring."""
        self.model.train()
        
        # Handle progressive unfreezing
        stage_updated = self.progressive_manager.update_stage(epoch, self.optimizer)
        if stage_updated or self.progressive_manager.is_enabled():
            stage_info = self.progressive_manager.get_current_stage_info()
            
            if stage_updated:
                logger.info(f"Progressive unfreezing: Advanced to stage {stage_info['stage_number']}/{stage_info['total_stages']}")
            
            if stage_info['stage'] == 'active':
                logger.debug(f"Current stage: {stage_info['stage_number']}/{stage_info['total_stages']}, "
                            f"frozen layers: {stage_info['frozen_layers']}, lr: {stage_info['lr']:.2e}")
        
        epoch_loss, num_batches, epoch_start_time = 0.0, 0, time.time()
        sample_embeddings1, sample_embeddings2, sample_labels = [], [], []
            
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.max_epochs}")
        
        for batch_idx, (img1, img2, labels) in enumerate(progress_bar):
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            emb1, emb2 = self._get_embeddings(img1, img2)
            
            # Collect sample embeddings for analysis
            if batch_idx < 5:
                sample_embeddings1.append(emb1.detach())
                sample_embeddings2.append(emb2.detach())
                sample_labels.append(labels.detach())
            
            # Calculate loss - with flag-based distillation handling
            if self._is_knowledge_distillation_enabled() and self.teacher_model is not None:
                # Knowledge distillation: pass images for teacher inference
                loss = self.criterion(emb1, emb2, labels, img1, img2)
            else:
                # Standard training: just use base loss
                loss = self.criterion(emb1, emb2, labels)
            
            loss.backward()
            
            # Log gradient info periodically
            if batch_idx % (self.cfg.log_interval * 5) == 0:
                self._log_gradient_info(epoch * len(self.train_loader) + batch_idx)
            

            grad_norm = clip_gradients(self.model, max_norm=1.0)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg Loss': f'{epoch_loss/num_batches:.4f}'})
            
            # Log batch metrics
            if batch_idx % self.cfg.log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                mlflow.log_metric("batch_loss", loss.item(), step=step)
                mlflow.log_metric("learning_rate", self.optimizer.param_groups[0]['lr'], step=step)
        
        # Compute embedding statistics
        if sample_embeddings1:
            all_emb1 = torch.cat(sample_embeddings1, dim=0)
            all_emb2 = torch.cat(sample_embeddings2, dim=0)
            all_labels = torch.cat(sample_labels, dim=0)
            self._compute_embedding_statistics(all_emb1, all_emb2, all_labels, epoch)
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches
        
        return avg_loss, epoch_time

    def _backup_config(self):
        """Backup configuration to file and MLflow."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_file = self.save_dir / f"config_backup_{timestamp}.yaml"
        
        try:
            with open(backup_file, 'w') as f:
                OmegaConf.save(self.cfg, f)
            mlflow.log_artifact(str(backup_file), "config")
            logger.info(f"Configuration backed up: {backup_file}")
        except Exception as e:
            logger.warning(f"Failed to backup config: {e}")


    def validate_epoch(self, epoch):
        """Runs a single validation epoch with detailed metrics."""
        avg_val_loss, val_metrics = self._run_evaluation(self.val_loader, epoch, "val")
        return avg_val_loss, val_metrics

    def final_evaluation(self):
        """Runs final evaluation on test set with comprehensive analysis."""
        logger.info("Running comprehensive final evaluation on test set...")
        avg_test_loss, test_metrics = self._run_evaluation(self.test_loader, epoch=-1, metric_prefix="test")
        
        logger.info(f"Final Test Loss: {avg_test_loss:.4f}")
        logger.info("Final Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric.replace('_', ' ').capitalize()}: {value:.4f}")

    def _run_evaluation(self, data_loader, epoch, metric_prefix):
            """Helper function to run evaluation on a given data loader."""
            self.model.eval()
            avg_loss = 0.0  # This will be validation loss OR test loss
            all_distances, all_labels = [], []
            all_embeddings1, all_embeddings2 = [], []

            with torch.no_grad():
                for img1, img2, labels in tqdm(data_loader, desc=f"{metric_prefix} Evaluation"):
                    img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                    
                    emb1, emb2 = self._get_embeddings(img1, img2)
                    
                    # Collect embeddings for visualization
                    if len(all_embeddings1) * emb1.shape[0] < 1000:
                        all_embeddings1.append(emb1.cpu())
                        all_embeddings2.append(emb2.cpu())
                    
                    # Calculate loss based on distillation flag
                    if self._is_knowledge_distillation_enabled() and self.teacher_model is not None:
                        # Use distillation loss for evaluation consistency
                        loss = self.criterion(emb1, emb2, labels, img1, img2)
                    else:
                        # Use base loss
                        loss = self.criterion(emb1, emb2, labels)
                    
                    avg_loss += loss.item()
                    
                    distances = self.criterion._compute_distance(emb1, emb2)
                    all_distances.extend(distances.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_loss /= len(data_loader)
            all_distances = np.array(all_distances)
            all_labels = np.array(all_labels)

            # Compute metrics
            metrics = MetricsCalculator.compute_comprehensive_metrics(all_distances, all_labels, epoch, metric_prefix)
            
            # Add loss to metrics - will be "val_loss" when called from validation, 
            # "test_loss" when called from test, but we use generic name for early stopping
            metrics[f'{metric_prefix}_loss'] = avg_loss
            
            # Create plots 
            if epoch >= 0:  # Only create plots during training, not final eval
                Plotter.plot_execution(self._create_validation_plots, all_distances, all_labels, epoch)
                if all_embeddings1:
                    Plotter.create_embedding_plots(all_embeddings1, all_embeddings2, all_labels, epoch, self.plots_dir)
            
            return avg_loss, metrics


    def _create_validation_plots(self, distances, labels, epoch):
        """Creates comprehensive validation plots."""
        try:
            distances = np.array(distances)
            labels = np.array(labels)
            scores = -distances
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Validation Metrics - Epoch {epoch+1}', fontsize=16)
            
            # ROC and PR curves, distributions, threshold analysis
            from sklearn.metrics import auc, roc_curve, precision_recall_curve
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
            axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            axes[0, 0].set_xlim([0.0, 1.0])
            axes[0, 0].set_ylim([0.0, 1.05])
            axes[0, 0].set_xlabel('False Positive Rate')
            axes[0, 0].set_ylabel('True Positive Rate')
            axes[0, 0].set_title('ROC Curve')
            axes[0, 0].legend(loc="lower right")
            axes[0, 0].grid(True, alpha=0.3)
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(labels, scores)
            pr_auc = auc(recall, precision)
            axes[0, 1].plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
            axes[0, 1].set_xlim([0.0, 1.0])
            axes[0, 1].set_ylim([0.0, 1.05])
            axes[0, 1].set_xlabel('Recall')
            axes[0, 1].set_ylabel('Precision')
            axes[0, 1].set_title('Precision-Recall Curve')
            axes[0, 1].legend(loc="lower left")
            axes[0, 1].grid(True, alpha=0.3)
            
            # Distance distributions
            same_distances = distances[labels == 1]
            diff_distances = distances[labels == 0]
            
            if len(same_distances) > 0 and len(diff_distances) > 0:
                axes[0, 2].hist(same_distances, bins=50, alpha=0.7, label='Same person', color='green', density=True)
                axes[0, 2].hist(diff_distances, bins=50, alpha=0.7, label='Different person', color='red', density=True)
                axes[0, 2].set_xlabel('Distance')
                axes[0, 2].set_ylabel('Density')
                axes[0, 2].set_title('Distance Distribution')
                axes[0, 2].legend()
                axes[0, 2].grid(True, alpha=0.3)
                
                # Score distributions
                same_scores = scores[labels == 1]
                diff_scores = scores[labels == 0]
                axes[1, 0].hist(same_scores, bins=50, alpha=0.7, label='Same person', color='green', density=True)
                axes[1, 0].hist(diff_scores, bins=50, alpha=0.7, label='Different person', color='red', density=True)
                axes[1, 0].set_xlabel('Score')
                axes[1, 0].set_ylabel('Density')
                axes[1, 0].set_title('Score Distribution')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Box plots
                axes[1, 1].boxplot([same_distances, diff_distances], labels=['Same', 'Different'])
                axes[1, 1].set_ylabel('Distance')
                axes[1, 1].set_title('Distance Box Plot')
                axes[1, 1].grid(True, alpha=0.3)
            
            # Threshold analysis
            thresholds = np.linspace(distances.min(), distances.max(), 100)
            accuracies = []
            for thresh in thresholds:
                preds = (distances <= thresh).astype(int)
                acc = np.mean(preds == labels)
                accuracies.append(acc)
            
            axes[1, 2].plot(thresholds, accuracies, 'b-', linewidth=2)
            axes[1, 2].set_xlabel('Threshold')
            axes[1, 2].set_ylabel('Accuracy')
            axes[1, 2].set_title('Accuracy vs Threshold')
            axes[1, 2].grid(True, alpha=0.3)
            
            # Mark optimal threshold
            best_idx = np.argmax(accuracies)
            best_threshold = thresholds[best_idx]
            best_accuracy = accuracies[best_idx]
            axes[1, 2].axvline(x=best_threshold, color='r', linestyle='--', alpha=0.5, 
                            label=f'Best: {best_accuracy:.3f} @ {best_threshold:.3f}')
            axes[1, 2].legend()
            
            plt.tight_layout()
            plot_path = self.plots_dir / f'validation_metrics_epoch_{epoch+1}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(str(plot_path))
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to create validation plots: {e}")
            plt.close('all')

    def _log_gradient_info(self, step):
        """Logs gradient information."""
        total_norm = 0
        param_count = 0
        max_grad = 0
        min_grad = float('inf')
        zero_grad_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
                
                max_grad = max(max_grad, param.grad.abs().max().item())
                min_grad = min(min_grad, param.grad.abs().min().item())
            elif param.requires_grad:
                zero_grad_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            
            # Log to MLflow
            mlflow.log_metric("grad_norm_total", total_norm, step=step)
            mlflow.log_metric("grad_max", max_grad, step=step)
            mlflow.log_metric("grad_min", min_grad, step=step)
            mlflow.log_metric("zero_grad_params", zero_grad_count, step=step)
            
            return total_norm
        return 0

    def _log_model_stats(self, epoch):
        """Logs model parameter statistics."""
        all_params = torch.cat([p.data.flatten() for p in self.model.parameters() if p.requires_grad])
        mlflow.log_metric("model_param_mean", all_params.mean().item(), step=epoch)
        mlflow.log_metric("model_param_std", all_params.std().item(), step=epoch)

    def _compute_embedding_statistics(self, embeddings1, embeddings2, labels, epoch):
        """Computes and logs embedding statistics."""
        emb1_np = embeddings1.cpu().detach().numpy()
        emb2_np = embeddings2.cpu().detach().numpy()
        labels_np = labels.cpu().numpy()
        
        norms1 = np.linalg.norm(emb1_np, axis=1)
        norms2 = np.linalg.norm(emb2_np, axis=1)
        emb1_norm = emb1_np / (norms1[:, np.newaxis] + 1e-8)
        emb2_norm = emb2_np / (norms2[:, np.newaxis] + 1e-8)
        cosine_sims = np.sum(emb1_norm * emb2_norm, axis=1)
        
        same_pairs = labels_np == 1
        diff_pairs = labels_np == 0
        
        mlflow.log_metric("emb_norm_mean", np.mean(np.concatenate([norms1, norms2])), step=epoch)
        mlflow.log_metric("emb_norm_std", np.std(np.concatenate([norms1, norms2])), step=epoch)
        
        if np.sum(same_pairs) > 0:
            mlflow.log_metric("cosine_sim_same_mean", np.mean(cosine_sims[same_pairs]), step=epoch)
            mlflow.log_metric("cosine_sim_same_std", np.std(cosine_sims[same_pairs]), step=epoch)
        
        if np.sum(diff_pairs) > 0:
            mlflow.log_metric("cosine_sim_diff_mean", np.mean(cosine_sims[diff_pairs]), step=epoch)
            mlflow.log_metric("cosine_sim_diff_std", np.std(cosine_sims[diff_pairs]), step=epoch)

    def save_checkpoint(self, epoch, train_loss, val_loss, val_metrics, is_best=False):
        """Saves a model checkpoint."""
        checkpoint_data = self.checkpoint_manager.create_checkpoint_data(
            self.model, self.optimizer, self.scheduler, epoch, train_loss,
            val_loss, val_metrics, self.cfg, self.training_history,
            self.best_score, self.epochs_without_improvement
        )
        
        # Add transfer learning info if applicable
        checkpoint_data = add_transfer_info_to_checkpoint(checkpoint_data, self.cfg)
        
        # Add knowledge distillation info if applicable
        checkpoint_data = add_distillation_info_to_checkpoint(checkpoint_data, self.cfg)
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth.tar"
        checkpoint_saved = self.checkpoint_manager.save_checkpoint_file(checkpoint_data, checkpoint_path)
        
        # Save best model if applicable
        if is_best and checkpoint_saved:
            best_path = self.save_dir / f"best_model_epoch_{epoch}.pth.tar"
            best_saved = self.checkpoint_manager.save_checkpoint_file(checkpoint_data, best_path)
            if best_saved:
                logger.info(f"New best model saved: {best_path}")

    def _get_model(self):
        """Initializes model from config."""
        model_name = self.cfg.model.name
        
        if model_name == "simple_cnn":
            model = SimpleCNN(self.cfg.model.params.embedding_dim)
        elif model_name == "lightweight_embedder":
            model = LightweightEmbedder(**self.cfg.model.params)
        elif model_name == "osnet_ain":
            variant_map = {
                "x1_0": osnet_ain_x1_0, "x0_75": osnet_ain_x0_75,
                "x0_5": osnet_ain_x0_5, "x0_25": osnet_ain_x0_25
            }
            
            model_fn = variant_map.get(self.cfg.model.params.variant)
            if not model_fn:
                raise ValueError(f"Unknown variant: {self.cfg.model.params.variant}")
            
            dropout_p = getattr(self.cfg.model.params, 'dropout_p', None)
            
            model = model_fn(
                num_classes=self.cfg.model.params.num_classes,
                pretrained=self.cfg.model.params.pretrained,
                loss='triplet',
                feature_dim=self.cfg.model.params.feature_dim,
                dropout_p=dropout_p
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
            
        return model.to(self.device)

    def _get_loss(self):
        """Initializes loss function from config."""
        if self.cfg.loss.name == "contrastive":
            return ContrastiveLoss(
                margin=self.cfg.loss.margin, 
                distance_metric=self.cfg.loss.distance_metric
            ).to(self.device)
        else:
            raise ValueError(f"Unknown loss: {self.cfg.loss.name}")

    def _get_optimizer(self):
        """Creates the optimizer."""
        return get_optimizer_with_differential_lr(self.model, self.cfg)

    def _get_scheduler(self):
        """Initializes learning rate scheduler from config."""
        return create_scheduler(self.optimizer, self.cfg)

    def _get_datasets(self):
        """Initializes datasets and dataloaders from config."""
        dataset_map = {
            "mnist_pairs": get_mnist_datasets,
            "generic_pairs": get_generic_datasets,
            "reid_pairs": get_reid_datasets
        }
        dataset_fn = dataset_map.get(self.cfg.dataset.name)
        if not dataset_fn:
            raise ValueError(f"Unknown dataset: {self.cfg.dataset.name}")
        return dataset_fn(self.cfg.dataset)

    def _setup_mlflow(self):
        """Sets up MLflow experiment and logs initial parameters."""
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)
        mlflow.start_run()
        logger.info("MLflow run started.")
        
        # Log Hydra config files
        hydra_dir = os.path.join(os.getcwd(), '.hydra')
        if os.path.exists(hydra_dir):
            mlflow.log_artifacts(hydra_dir, "config_files")
            
        # Base parameters
        params = {
            "model_name": self.cfg.model.name,
            "dataset_name": self.cfg.dataset.name,
            "learning_rate": self.cfg.learning_rate,
            "weight_decay": self.cfg.weight_decay,
            "max_epochs": self.cfg.max_epochs,
            "batch_size": self.cfg.dataset.batch_size,
            "optimizer": "adam",
            "loss_function": self.cfg.loss.name,
            "loss_margin": self.cfg.loss.margin,
            "distance_metric": self.cfg.loss.distance_metric
        }
        
        # Add knowledge distillation parameters if enabled
        if self._is_knowledge_distillation_enabled():
            params.update({
                "knowledge_distillation_enabled": True,
                "teacher_variant": self.cfg.knowledge_distillation.get('teacher_variant', 'x1_0'),
                "distillation_temperature": self.cfg.knowledge_distillation.get('temperature', 4.0),
                "distillation_alpha": self.cfg.knowledge_distillation.get('alpha', 0.7),
                "teacher_checkpoint": self.cfg.knowledge_distillation.teacher_checkpoint,
                "feature_matching_weight": self.cfg.knowledge_distillation.get('feature_weight', 0.1)
            })
        else:
            params["knowledge_distillation_enabled"] = False
        
        mlflow.log_params(params)
        
        if hasattr(self.cfg.model.params, 'variant'):
            mlflow.log_param("model_variant", self.cfg.model.params.variant)
        if hasattr(self.cfg.model.params, 'feature_dim'):
            mlflow.log_param("feature_dim", self.cfg.model.params.feature_dim)
        if hasattr(self.cfg, 'transfer_learning') and self.cfg.transfer_learning.enabled:
            mlflow.log_param("transfer_learning", True)
            mlflow.log_param("tl_source_variant", self.cfg.transfer_learning.source_variant)
            mlflow.log_param("tl_method", self.cfg.transfer_learning.method)

        self._backup_config()

    def _get_embeddings(self, img1, img2):
        """Extracts embeddings from the model."""
        if hasattr(self.model, "forward_one"):
            return self.model(img1, img2)
        
        output1, output2 = self.model(img1), self.model(img2)
        if isinstance(output1, tuple):
            _, emb1 = output1
            _, emb2 = output2
        else:
            emb1, emb2 = output1, output2
        return emb1, emb2


@hydra.main(config_path="../configs", config_name="transfer_learning_config", version_base=None)
def main(cfg: DictConfig):
    """Main training function with enhanced monitoring"""
    logger.info("="*60)
    logger.info("ENHANCED REID TRAINING WITH COMPREHENSIVE MONITORING")
    logger.info("="*60)
    logger.info(f"Model: {cfg.model.name}, Variant: {cfg.model.params.variant}")
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Learning Rate: {cfg.learning_rate}, Max Epochs: {cfg.max_epochs}")
    logger.info("="*60)
    
    try:
        trainer = EnhancedReIDTrainer(cfg)
        trainer.train()
    except Exception as e:
        logger.exception("Training failed with an unexpected error.")
        mlflow.end_run(status="FAILED")

if __name__ == "__main__":
    main()