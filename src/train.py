#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import auc, roc_curve, precision_recall_curve, confusion_matrix
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
import hydra
from omegaconf import DictConfig
import mlflow
import mlflow.pytorch
import os
from typing import Tuple
from pathlib import Path
import time
import logging

from models.osnet_ain import osnet_ain_x1_0, osnet_ain_x0_75, osnet_ain_x0_5, osnet_ain_x0_25
from datasets.generic_dataset import get_generic_datasets
from datasets.mnist_dataset import get_mnist_datasets
from datasets.reid_dataset import get_reid_datasets
from losses.contrastive_loss import ContrastiveLoss
from models.lightweight_embedder import LightweightEmbedder
from models.simple_cnn import SimpleCNN
from utils.transfer_learning import apply_transfer_learning, freeze_layers, get_optimizer_with_differential_lr, add_transfer_info_to_checkpoint

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedReIDTrainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        self.device = torch.device("cuda" if torch.cuda.is_available() 
                                 else "mps" if torch.backends.mps.is_available() 
                                 else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = self._get_model()
        
        self.model = apply_transfer_learning(self.model, cfg, self.device)
        freeze_layers(self.model, cfg)
        
        self.criterion = self._get_loss()
        self.optimizer = self._get_optimizer()
        self.scheduler = self._get_scheduler()
        
        self.train_loader, self.val_loader, self.test_loader = self._get_datasets()
        
        self._setup_mlflow()
        
        self.save_dir = Path(cfg.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.training_history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_accuracy': [],
            'learning_rate': [], 'grad_norm': [], 'epoch_time': []
        }
        
        self.plots_dir = self.save_dir / "plots"
        self.plots_dir.mkdir(exist_ok=True)

        self.eval_interval = self.cfg.get('evaluation_interval', 1) 
        self.checkpoint_interval = self.cfg.get('checkpoint_epoch', 1)
        
    
        self.patience = self.cfg.early_stopping.patience
        self.monitor_metric = self.cfg.early_stopping.monitor
        self.mode = self.cfg.early_stopping.mode
        self.min_delta = self.cfg.early_stopping.min_delta
        
    def train(self):
        logger.info("Starting enhanced training with comprehensive monitoring...")
        logger.info(f"Model: {self.cfg.model.name}, Variant: {self.cfg.model.params.variant}")
        logger.info(f"Dataset: {self.cfg.dataset.name}")
        logger.info(f"Learning Rate: {self.cfg.learning_rate}, Max Epochs: {self.cfg.max_epochs}")
        logger.info("="*60)
        

        self.best_score = -float('inf') if self.mode == 'max' else float('inf')
        self.epochs_without_improvement = 0
        
        for epoch in range(self.cfg.max_epochs):
            train_loss, epoch_time = self.train_epoch(epoch)
            early_stopping = False
            
            if (epoch + 1) % self.eval_interval == 0:
                val_loss, val_metrics = self.validate_epoch(epoch)
                
                self._log_model_stats(epoch)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                if self.scheduler:
                    self.scheduler.step(val_loss if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)
                
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_auc'].append(val_metrics['auc'])
                self.training_history['val_accuracy'].append(val_metrics['accuracy'])
                mlflow.log_metric("epoch_val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_auc", val_metrics['auc'], step=epoch)
                mlflow.log_metric("val_accuracy", val_metrics['accuracy'], step=epoch)
                
                is_best, early_stopping = self._check_early_stopping(val_metrics)
                logger.info(f"Epoch {epoch+1}/{self.cfg.max_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_metrics['auc']:.4f}, Val F1: {val_metrics['f1']:.4f}, Epoch Time: {epoch_time:.1f}s, LR: {current_lr:.6f}")
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{self.cfg.max_epochs} - Train Loss: {train_loss:.4f}, Epoch Time: {epoch_time:.1f}s, LR: {current_lr:.6f}")
                
            self.training_history['epoch'].append(epoch)
            self.training_history['train_loss'].append(train_loss)
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['epoch_time'].append(epoch_time)
            mlflow.log_metric("epoch_train_loss", train_loss, step=epoch)
            mlflow.log_metric("epoch_time", epoch_time, step=epoch)
            
            if (epoch + 1) % self.checkpoint_interval == 0 or early_stopping:
                self.save_checkpoint(epoch, train_loss, val_loss, val_metrics, is_best)
                
            if early_stopping:
                logger.info(f"Early stopping triggered! No improvement in '{self.monitor_metric}' for {self.patience} epochs. Stopping training.")
                self._create_training_history_plot()
                break
        
        logger.info("Training completed!")
        logger.info(f"Best validation score for '{self.monitor_metric}': {self.best_score:.4f}")
        
        if self.cfg.run_eval: 
            self.final_evaluation()
        
        mlflow.end_run()

    def train_epoch(self, epoch):
        """Runs a single training epoch with detailed monitoring."""
        self.model.train()
        epoch_loss, num_batches, epoch_start_time = 0.0, 0, time.time()
        
        sample_embeddings1, sample_embeddings2, sample_labels = [], [], []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.cfg.max_epochs}")
        
        for batch_idx, (img1, img2, labels) in enumerate(progress_bar):
            img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            emb1, emb2 = self._get_embeddings(img1, img2)
            
            if batch_idx < 5:
                sample_embeddings1.append(emb1.detach())
                sample_embeddings2.append(emb2.detach())
                sample_labels.append(labels.detach())
            
            loss = self.criterion(emb1, emb2, labels)
            loss.backward()
            
            if batch_idx % (self.cfg.log_interval * 5) == 0:
                self._log_gradient_info(epoch * len(self.train_loader) + batch_idx)
            
            self.optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg Loss': f'{epoch_loss/num_batches:.4f}'})
            
            if batch_idx % self.cfg.log_interval == 0:
                step = epoch * len(self.train_loader) + batch_idx
                mlflow.log_metric("batch_loss", loss.item(), step=step)
                mlflow.log_metric("learning_rate", self.optimizer.param_groups[0]['lr'], step=step)
        
        if sample_embeddings1:
            all_emb1, all_emb2, all_labels = torch.cat(sample_embeddings1, dim=0), torch.cat(sample_embeddings2, dim=0), torch.cat(sample_labels, dim=0)
            self._compute_embedding_statistics(all_emb1, all_emb2, all_labels, epoch)
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = epoch_loss / num_batches
        
        return avg_loss, epoch_time

    def validate_epoch(self, epoch):
        """Runs a single validation epoch with detailed metrics."""
        avg_val_loss, val_metrics = self._run_evaluation(self.val_loader, epoch, "val")
        
        # Log validation loss separately
        mlflow.log_metric("epoch_val_loss", avg_val_loss, step=epoch)
        
        return avg_val_loss, val_metrics

    def final_evaluation(self):
        """Runs final evaluation on test set with comprehensive analysis."""
        logger.info("Running comprehensive final evaluation on test set...")
        
        # Note: You can pass a different epoch value for final eval, e.g., -1
        # to indicate a final run outside of the training loop.
        avg_test_loss, test_metrics = self._run_evaluation(self.test_loader, epoch=-1, metric_prefix="test")
        logger.info(f"Final Test Loss: {avg_test_loss:.4f}")

        # The helper function already logs the metrics with the "test_" prefix.
        logger.info("Final Test Results:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric.replace('_', ' ').capitalize()}: {value:.4f}")
                
    def _log_gradient_info(self, epoch):
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
                
                # Track max/min gradients
                max_grad = max(max_grad, param.grad.abs().max().item())
                min_grad = min(min_grad, param.grad.abs().min().item())
            elif param.requires_grad:
                zero_grad_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            
            # Log to MLflow
            mlflow.log_metric("grad_norm_total", total_norm, step=epoch)
            mlflow.log_metric("grad_max", max_grad, step=epoch)
            mlflow.log_metric("grad_min", min_grad, step=epoch)
            mlflow.log_metric("zero_grad_params", zero_grad_count, step=epoch)
            
            return total_norm
        return 0

    def _log_model_stats(self, epoch):
        """Logs model parameter statistics."""
        all_params = torch.cat([p.data.flatten() for p in self.model.parameters() if p.requires_grad])
        mlflow.log_metric("model_param_mean", all_params.mean().item(), step=epoch)
        mlflow.log_metric("model_param_std", all_params.std().item(), step=epoch)
        logger.debug(f"Model parameters at epoch {epoch} - Mean: {all_params.mean().item():.4f}, Std: {all_params.std().item():.4f}")

    def _compute_embedding_statistics(self, embeddings1, embeddings2, labels, epoch):
        """Computes and logs embedding statistics."""
        emb1_np, emb2_np, labels_np = embeddings1.cpu().detach().numpy(), embeddings2.cpu().detach().numpy(), labels.cpu().numpy()
        
        norms1, norms2 = np.linalg.norm(emb1_np, axis=1), np.linalg.norm(emb2_np, axis=1)
        emb1_norm, emb2_norm = emb1_np / (norms1[:, np.newaxis] + 1e-8), emb2_np / (norms2[:, np.newaxis] + 1e-8)
        cosine_sims = np.sum(emb1_norm * emb2_norm, axis=1)
        
        same_pairs, diff_pairs = labels_np == 1, labels_np == 0
        
        mlflow.log_metric("emb_norm_mean", np.mean(np.concatenate([norms1, norms2])), step=epoch)
        mlflow.log_metric("emb_norm_std", np.std(np.concatenate([norms1, norms2])), step=epoch)
        
        if np.sum(same_pairs) > 0:
            mlflow.log_metric("cosine_sim_same_mean", np.mean(cosine_sims[same_pairs]), step=epoch)
            mlflow.log_metric("cosine_sim_same_std", np.std(cosine_sims[same_pairs]), step=epoch)
        
        if np.sum(diff_pairs) > 0:
            mlflow.log_metric("cosine_sim_diff_mean", np.mean(cosine_sims[diff_pairs]), step=epoch)
            mlflow.log_metric("cosine_sim_diff_std", np.std(cosine_sims[diff_pairs]), step=epoch)
        
        logger.debug(f"Embedding stats at epoch {epoch}: same_mean={np.mean(cosine_sims[same_pairs]):.4f}, diff_mean={np.mean(cosine_sims[diff_pairs]):.4f}")
        
    def _compute_comprehensive_metrics(self, distances, labels, epoch):
        """Computes comprehensive validation metrics."""
        scores = -distances
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold, optimal_tpr, optimal_fpr = -roc_thresholds[optimal_idx], tpr[optimal_idx], fpr[optimal_idx]
        
        predictions = (distances <= optimal_threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = fpr[eer_idx]
        
        mlflow.log_metric("val_pr_auc", pr_auc, step=epoch)
        mlflow.log_metric("val_precision", precision_score, step=epoch)
        mlflow.log_metric("val_recall", recall_score, step=epoch)
        mlflow.log_metric("val_f1", f1_score, step=epoch)
        mlflow.log_metric("val_eer", eer, step=epoch)
        mlflow.log_metric("val_optimal_tpr", optimal_tpr, step=epoch)
        mlflow.log_metric("val_optimal_fpr", optimal_fpr, step=epoch)
        
        return {
            'auc': roc_auc, 'pr_auc': pr_auc, 'accuracy': accuracy,
            'precision': precision_score, 'recall': recall_score, 'f1': f1_score,
            'eer': eer, 'optimal_threshold': optimal_threshold
        }    
        
    def _create_embedding_plots(self, embeddings1, embeddings2, labels, epoch):
        """Creates embedding visualization plots."""
        # Concatenate embeddings if they're in a list
        if isinstance(embeddings1, list) and len(embeddings1) > 0:
            all_embeddings = torch.cat(embeddings1, dim=0)
        else:
            all_embeddings = embeddings1
        
        if all_embeddings is None or len(all_embeddings) == 0:
            logger.warning("No embeddings available for visualization")
            return
        
        # Convert to numpy
        embeddings_np = all_embeddings.numpy() if isinstance(all_embeddings, torch.Tensor) else all_embeddings
        
        # Limit the number of samples for visualization
        max_samples = min(1000, len(embeddings_np))
        embeddings_np = embeddings_np[:max_samples]
        
        # Create labels for visualization if not provided or if wrong shape
        if labels is None or (isinstance(labels, np.ndarray) and len(labels) != len(embeddings_np)):
            # Create pseudo-labels based on embedding index for coloring
            labels_np = np.arange(len(embeddings_np)) % 10  # Creates groups for visualization
        else:
            labels_np = labels[:len(embeddings_np)] if isinstance(labels, np.ndarray) else np.array(labels)[:len(embeddings_np)]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Embedding Visualizations - Epoch {epoch+1}', fontsize=16)
        
        # PCA visualization
        if embeddings_np.shape[1] > 2:
            try:
                pca = PCA(n_components=2)
                pca_embeddings = pca.fit_transform(embeddings_np)
                scatter = axes[0].scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], 
                                        c=labels_np, cmap='viridis', alpha=0.6)
                axes[0].set_title(f'PCA Embedding Space\n(Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
                axes[0].set_xlabel('PC1')
                axes[0].set_ylabel('PC2')
                plt.colorbar(scatter, ax=axes[0])
            except Exception as e:
                logger.warning(f"PCA visualization failed: {e}")
                axes[0].text(0.5, 0.5, 'PCA visualization failed', 
                        ha='center', va='center', transform=axes[0].transAxes)
        
        # t-SNE visualization
        if len(embeddings_np) > 50:
            try:
                perplexity = min(30, max(5, len(embeddings_np) // 4))
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                tsne_embeddings = tsne.fit_transform(embeddings_np)
                scatter = axes[1].scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                        c=labels_np, cmap='viridis', alpha=0.6)
                axes[1].set_title('t-SNE Embedding Space')
                axes[1].set_xlabel('t-SNE 1')
                axes[1].set_ylabel('t-SNE 2')
                plt.colorbar(scatter, ax=axes[1])
            except Exception as e:
                logger.warning(f"t-SNE visualization failed: {e}")
                axes[1].text(0.5, 0.5, 'Not enough samples\nfor t-SNE', 
                        ha='center', va='center', transform=axes[1].transAxes)
        else:
            axes[1].text(0.5, 0.5, f'Not enough samples for t-SNE\n({len(embeddings_np)} samples)', 
                    ha='center', va='center', transform=axes[1].transAxes)
        
        # Embedding norm distribution
        norms = np.linalg.norm(embeddings_np, axis=1)
        axes[2].hist(norms, bins=30, alpha=0.7, color='blue', density=True)
        axes[2].set_xlabel('Embedding Norm')
        axes[2].set_ylabel('Density')
        axes[2].set_title(f'Embedding Norm Distribution\n(Mean: {norms.mean():.3f}, Std: {norms.std():.3f})')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / f'embedding_visualization_epoch_{epoch+1}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
        logger.debug(f"Saved embedding visualization to {plot_path}")

    def _create_validation_plots(self, distances, labels, epoch):
        """Creates comprehensive validation plots."""
        distances = np.array(distances)
        labels = np.array(labels)
        scores = -distances
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Validation Metrics - Epoch {epoch+1}', fontsize=16)
        
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
        else:
            logger.warning("Not enough data for distribution plots")
        
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
        
        # Find and mark the optimal threshold
        best_idx = np.argmax(accuracies)
        best_threshold = thresholds[best_idx]
        best_accuracy = accuracies[best_idx]
        axes[1, 2].axvline(x=best_threshold, color='r', linestyle='--', alpha=0.5, 
                        label=f'Best: {best_accuracy:.3f} @ {best_threshold:.3f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        
        # Save and log to MLflow
        plot_path = self.plots_dir / f'validation_metrics_epoch_{epoch+1}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
        logger.debug(f"Saved validation plots to {plot_path}")


    def _create_training_history_plot(self):
        """Creates training history plots."""
        if len(self.training_history['epoch']) < 2: return
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Training History', fontsize=16)
        epochs = self.training_history['epoch']
        
        axes[0, 0].plot(epochs, self.training_history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.training_history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        
        axes[0, 1].plot(epochs, self.training_history['val_auc'], 'g-', label='Val AUC')
        axes[0, 1].plot(epochs, self.training_history['val_accuracy'], 'b-', label='Val Accuracy')
        axes[0, 1].set_title('Validation Metrics')
        
        axes[0, 2].plot(epochs, self.training_history['learning_rate'], 'purple', linewidth=2)
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_yscale('log')
        
        if self.training_history['grad_norm']:
            axes[1, 0].plot(epochs, self.training_history['grad_norm'], 'orange', linewidth=2)
            axes[1, 0].set_title('Gradient Norm')
        
        axes[1, 1].plot(epochs, self.training_history['epoch_time'], 'brown', linewidth=2)
        axes[1, 1].set_title('Epoch Training Time')
        
        if len(self.training_history['train_loss']) > 0 and len(self.training_history['val_loss']) > 0:
            loss_ratio = np.array(self.training_history['val_loss']) / (np.array(self.training_history['train_loss']) + 1e-8)
            axes[1, 2].plot(epochs, loss_ratio, 'red', linewidth=2)
            axes[1, 2].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
            axes[1, 2].set_title('Overfitting Indicator')
        
        for ax in axes.flatten():
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.plots_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        mlflow.log_artifact(str(plot_path))
        plt.close()
        
    def _check_early_stopping(self, val_metrics: dict) -> Tuple[bool, bool]:        
        """
        Handles the logic for early stopping based on validation metrics.
        
        Returns:
            A tuple of (is_best, should_stop) booleans.
        """
        
        current_score = val_metrics.get(self.monitor_metric, float('nan'))
        is_best = False
        
        if self.mode == 'max':
            if current_score > self.best_score + self.min_delta:
                self.best_score = current_score
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1
        else: # mode is 'min'
            if current_score < self.best_score - self.min_delta:
                self.best_score = current_score
                self.epochs_without_improvement = 0
                is_best = True
            else:
                self.epochs_without_improvement += 1

        if is_best:
            logger.info(f"New best score ({self.monitor_metric}: {self.best_score:.4f}) found. Resetting early stopping counter.")
        
        should_stop = self.epochs_without_improvement >= self.patience
        return is_best, should_stop
    
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
                logger.error(f"Unknown variant: {self.cfg.model.params.variant}")
                raise ValueError(f"Unknown variant: {self.cfg.model.params.variant}")
            
            model = model_fn(
                num_classes=self.cfg.model.params.num_classes,
                pretrained=self.cfg.model.params.pretrained,
                loss='triplet',
                feature_dim=self.cfg.model.params.feature_dim
            )
        else:
            logger.error(f"Unknown model: {model_name}")
            raise ValueError(f"Unknown model: {model_name}")
            
        return model.to(self.device)

    def _get_loss(self):
        """Initializes loss function from config."""
        if self.cfg.loss.name == "contrastive":
            return ContrastiveLoss(margin=self.cfg.loss.margin, distance_metric=self.cfg.loss.distance_metric).to(self.device)
        else:
            logger.error(f"Unknown loss: {self.cfg.loss.name}")
            raise ValueError(f"Unknown loss: {self.cfg.loss.name}")

    def _get_optimizer(self):
        """Creates the optimizer, using helper for transfer learning logic."""
        return get_optimizer_with_differential_lr(self.model, self.cfg)
        
        
    def _get_scheduler(self):
        """Initializes learning rate scheduler from config."""
        if not hasattr(self.cfg, 'scheduler') or not self.cfg.scheduler: return None
        if self.cfg.scheduler.name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.cfg.scheduler.step_size, gamma=self.cfg.scheduler.gamma)
        elif self.cfg.scheduler.name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg.max_epochs)
        elif self.cfg.scheduler.name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=self.cfg.scheduler.patience, factor=self.cfg.scheduler.gamma)
        else: return None

    def _get_datasets(self):
        """Initializes datasets and dataloaders from config."""
        dataset_map = {
            "mnist_pairs": get_mnist_datasets, "generic_pairs": get_generic_datasets, "reid_pairs": get_reid_datasets
        }
        dataset_fn = dataset_map.get(self.cfg.dataset.name)
        if not dataset_fn:
            logger.error(f"Unknown dataset: {self.cfg.dataset.name}")
            raise ValueError(f"Unknown dataset: {self.cfg.dataset.name}")
        return dataset_fn(self.cfg.dataset)

    def _setup_mlflow(self):
        """Sets up MLflow experiment and logs initial parameters."""
        mlflow.set_experiment(self.cfg.mlflow.experiment_name)
        mlflow.start_run()
        logger.info("MLflow run started.")
        
        # Get the path to the .hydra directory created by Hydra
        hydra_dir = os.path.join(os.getcwd(), '.hydra')
        if os.path.exists(hydra_dir):
            mlflow.log_artifacts(hydra_dir, "config_files")
            logger.info(f"Logged Hydra configuration files from {hydra_dir}.")
            
        mlflow.log_params({
            "model_name": self.cfg.model.name, "dataset_name": self.cfg.dataset.name,
            "learning_rate": self.cfg.learning_rate, "weight_decay": self.cfg.weight_decay,
            "max_epochs": self.cfg.max_epochs, "batch_size": self.cfg.dataset.batch_size,
            "optimizer": "adam", "loss_function": self.cfg.loss.name,
            "loss_margin": self.cfg.loss.margin, "distance_metric": self.cfg.loss.distance_metric
        })
        if hasattr(self.cfg.model.params, 'variant'): mlflow.log_param("model_variant", self.cfg.model.params.variant)
        if hasattr(self.cfg.model.params, 'feature_dim'): mlflow.log_param("feature_dim", self.cfg.model.params.feature_dim)
        if hasattr(self.cfg, 'transfer_learning') and self.cfg.transfer_learning.enabled:
            mlflow.log_param("transfer_learning", True)
            mlflow.log_param("tl_source_variant", self.cfg.transfer_learning.source_variant)
            mlflow.log_param("tl_method", self.cfg.transfer_learning.method)

    def _get_embeddings(self, img1, img2):
        """Extracts embeddings from the model."""
        if hasattr(self.model, "forward_one"): return self.model(img1, img2)
        output1, output2 = self.model(img1), self.model(img2)
        if isinstance(output1, tuple): _, emb1 = output1; _, emb2 = output2
        else: emb1, emb2 = output1, output2
        return emb1, emb2


    def save_checkpoint(self, epoch, train_loss, val_loss, val_metrics, is_best=False):
        """Saves a model checkpoint."""
        
        checkpoint = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': self.cfg,
            'training_history': self.training_history
        }
        
        # Add transfer learning info if applicable
        checkpoint = add_transfer_info_to_checkpoint(checkpoint, self.cfg)
        
        if self.scheduler:
            checkpoint['scheduler'] = self.scheduler.state_dict()
        
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pth.tar"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        if is_best:
            best_path = self.save_dir / f"best_model_epoch_{epoch}.pth.tar"
            torch.save(checkpoint, best_path)
            mlflow.log_artifact(str(best_path))
            logger.info(f"New best model saved: {best_path}")
        
        mlflow.log_artifact(str(checkpoint_path))


    def _run_evaluation(self, data_loader, epoch, metric_prefix):
            """Helper function to run evaluation on a given data loader."""
            self.model.eval()
            avg_loss = 0.0
            all_distances, all_labels = [], []
            all_embeddings1, all_embeddings2 = [], []

            with torch.no_grad():
                for img1, img2, labels in tqdm(data_loader, desc=f"{metric_prefix} Evaluation"):
                    img1, img2, labels = img1.to(self.device), img2.to(self.device), labels.to(self.device)
                    
                    emb1, emb2 = self._get_embeddings(img1, img2)
                    
                    if len(all_embeddings1) * emb1.shape[0] < 1000:
                        all_embeddings1.append(emb1.cpu())
                        all_embeddings2.append(emb2.cpu())
                    
                    loss = self.criterion(emb1, emb2, labels)
                    avg_loss += loss.item()
                    
                    distances = self.criterion._compute_distance(emb1, emb2)
                    all_distances.extend(distances.cpu().numpy().flatten())
                    all_labels.extend(labels.cpu().numpy())
            
            avg_loss /= len(data_loader)
            
            all_distances = np.array(all_distances)
            all_labels = np.array(all_labels)

            # Compute metrics
            metrics = self._compute_comprehensive_metrics(all_distances, all_labels, epoch, metric_prefix)
            
            # Create plots
            self._create_validation_plots(all_distances, all_labels, epoch)
            if all_embeddings1:
                self._create_embedding_plots(all_embeddings1, all_embeddings2, all_labels, epoch)
                
            return avg_loss, metrics

    def _compute_comprehensive_metrics(self, distances, labels, epoch, metric_prefix):
        """Computes comprehensive validation metrics."""
        scores = -distances
        fpr, tpr, roc_thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, pr_thresholds = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)
        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold, optimal_tpr, optimal_fpr = -roc_thresholds[optimal_idx], tpr[optimal_idx], fpr[optimal_idx]
        
        predictions = (distances <= optimal_threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision_score * recall_score) / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        
        eer_idx = np.argmin(np.abs(fpr - (1 - tpr)))
        eer = fpr[eer_idx]

        # Log metrics with dynamic prefix
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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
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
