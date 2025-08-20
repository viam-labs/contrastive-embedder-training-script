#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import mlflow
import logging

logger = logging.getLogger(__name__)

class Plotter:
    """Handles plotting with error recovery."""
    
    @staticmethod
    def plot_execution(plot_function, *args, **kwargs):
        """Execute plotting function with error handling."""
        try:
            result = plot_function(*args, **kwargs)
            return result if result is not None else True
        except Exception as e:
            function_name = plot_function.__name__.replace('_create_', '').replace('_', ' ')
            logger.warning(f"Failed to create {function_name}: {e}")
            plt.close('all')
            return False
    
    @staticmethod
    def create_training_history_plot(training_history, plots_dir):
        """Creates training history plots with error handling."""
        try:
            if len(training_history['epoch']) < 2: 
                logger.warning("Not enough epochs for training history plot")
                return False
                
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Training History', fontsize=16)
            epochs = training_history['epoch']
            
            # Plot training loss for all epochs
            axes[0, 0].plot(epochs, training_history['train_loss'], 'b-', label='Train Loss')
            
            # Plot validation loss only for epochs where validation occurred
            if len(training_history['val_epochs']) > 0 and len(training_history['val_loss']) > 0:
                val_epochs = training_history['val_epochs']
                axes[0, 0].plot(val_epochs, training_history['val_loss'], 'r-', label='Val Loss')
            
            axes[0, 0].set_title('Training and Validation Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Validation metrics
            if len(training_history['val_epochs']) > 0:
                val_epochs = training_history['val_epochs']
                axes[0, 1].plot(val_epochs, training_history['val_auc'], 'g-', label='Val AUC')
                axes[0, 1].plot(val_epochs, training_history['val_accuracy'], 'b-', label='Val Accuracy')
            axes[0, 1].set_title('Validation Metrics')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Learning rate
            axes[0, 2].plot(epochs, training_history['learning_rate'], 'purple', linewidth=2)
            axes[0, 2].set_title('Learning Rate Schedule')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True, alpha=0.3)
            
            # Gradient norm (if available)
            if 'grad_norm' in training_history and len(training_history['grad_norm']) > 0:
                grad_epochs = epochs[:len(training_history['grad_norm'])]
                axes[1, 0].plot(grad_epochs, training_history['grad_norm'], 'orange', linewidth=2)
                axes[1, 0].set_title('Gradient Norm')
                axes[1, 0].grid(True, alpha=0.3)
            else:
                axes[1, 0].text(0.5, 0.5, 'Gradient norm data\nnot available', 
                            ha='center', va='center', transform=axes[1, 0].transAxes)
                axes[1, 0].set_title('Gradient Norm')
            
            # Epoch time
            axes[1, 1].plot(epochs, training_history['epoch_time'], 'brown', linewidth=2)
            axes[1, 1].set_title('Epoch Training Time')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Overfitting indicator
            Plotter._create_overfitting_plot(axes[1, 2], training_history)
            
            plt.tight_layout()
            plot_path = plots_dir / 'training_history.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(str(plot_path))
            plt.close()
            logger.info(f"Training history plot saved to {plot_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to create training history plot: {e}")
            plt.close('all')
            return False
    
    @staticmethod
    def _create_overfitting_plot(ax, training_history):
        """Create overfitting indicator subplot."""
        try:
            if (len(training_history['val_loss']) > 0 and 
                len(training_history['train_loss']) > 0 and
                len(training_history['val_epochs']) > 0):
                
                val_epochs = training_history['val_epochs']
                train_losses_at_val = [training_history['train_loss'][epoch] 
                                     for epoch in val_epochs if epoch < len(training_history['train_loss'])]
                
                if len(train_losses_at_val) == len(training_history['val_loss']):
                    loss_ratio = np.array(training_history['val_loss']) / (np.array(train_losses_at_val) + 1e-8)
                    ax.plot(val_epochs, loss_ratio, 'red', linewidth=2)
                    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
                    ax.set_title('Overfitting Indicator (Val/Train Loss)')
                    ax.grid(True, alpha=0.3)
                    return
            
            ax.text(0.5, 0.5, 'Overfitting indicator\nnot available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Overfitting Indicator')
        except Exception:
            ax.text(0.5, 0.5, 'Overfitting indicator\nfailed to create', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Overfitting Indicator')
    
    @staticmethod
    def create_embedding_plots(embeddings1, embeddings2, labels, epoch, plots_dir):
        """Creates embedding visualization plots with error handling."""
        try:
            if isinstance(embeddings1, list) and len(embeddings1) > 0:
                all_embeddings = torch.cat(embeddings1, dim=0)
            else:
                all_embeddings = embeddings1
            
            if all_embeddings is None or len(all_embeddings) == 0:
                logger.warning("No embeddings available for visualization")
                return False
            
            embeddings_np = all_embeddings.numpy() if isinstance(all_embeddings, torch.Tensor) else all_embeddings
            max_samples = min(1000, len(embeddings_np))
            embeddings_np = embeddings_np[:max_samples]
            
            if labels is None or (isinstance(labels, np.ndarray) and len(labels) != len(embeddings_np)):
                labels_np = np.arange(len(embeddings_np)) % 10
            else:
                labels_np = labels[:len(embeddings_np)] if isinstance(labels, np.ndarray) else np.array(labels)[:len(embeddings_np)]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Embedding Visualizations - Epoch {epoch+1}', fontsize=16)
            
            # PCA visualization
            Plotter._create_pca_plot(axes[0], embeddings_np, labels_np)
            
            # t-SNE visualization
            Plotter._create_tsne_plot(axes[1], embeddings_np, labels_np)
            
            # Embedding norm distribution
            Plotter._create_norm_distribution_plot(axes[2], embeddings_np)
            
            plt.tight_layout()
            plot_path = plots_dir / f'embedding_visualization_epoch_{epoch+1}.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(str(plot_path))
            plt.close()
            logger.debug(f"Saved embedding visualization to {plot_path}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to create embedding plots for epoch {epoch}: {e}")
            plt.close('all')
            return False
    
    @staticmethod
    def _create_pca_plot(ax, embeddings_np, labels_np):
        """Create PCA subplot."""
        try:
            if embeddings_np.shape[1] > 2:
                pca = PCA(n_components=2)
                pca_embeddings = pca.fit_transform(embeddings_np)
                scatter = ax.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1], 
                                   c=labels_np, cmap='viridis', alpha=0.6)
                ax.set_title(f'PCA Embedding Space\n(Explained Variance: {pca.explained_variance_ratio_.sum():.3f})')
                ax.set_xlabel('PC1')
                ax.set_ylabel('PC2')
                plt.colorbar(scatter, ax=ax)
        except Exception as e:
            logger.warning(f"PCA visualization failed: {e}")
            ax.text(0.5, 0.5, 'PCA visualization failed', 
                   ha='center', va='center', transform=ax.transAxes)
    
    @staticmethod
    def _create_tsne_plot(ax, embeddings_np, labels_np):
        """Create t-SNE subplot."""
        try:
            if len(embeddings_np) > 50:
                perplexity = min(30, max(5, len(embeddings_np) // 4))
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                tsne_embeddings = tsne.fit_transform(embeddings_np)
                scatter = ax.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1], 
                                   c=labels_np, cmap='viridis', alpha=0.6)
                ax.set_title('t-SNE Embedding Space')
                ax.set_xlabel('t-SNE 1')
                ax.set_ylabel('t-SNE 2')
                plt.colorbar(scatter, ax=ax)
            else:
                ax.text(0.5, 0.5, f'Not enough samples for t-SNE\n({len(embeddings_np)} samples)', 
                       ha='center', va='center', transform=ax.transAxes)
        except Exception as e:
            logger.warning(f"t-SNE visualization failed: {e}")
            ax.text(0.5, 0.5, 'Not enough samples\nfor t-SNE', 
                   ha='center', va='center', transform=ax.transAxes)
    
    @staticmethod
    def _create_norm_distribution_plot(ax, embeddings_np):
        """Create embedding norm distribution subplot."""
        try:
            norms = np.linalg.norm(embeddings_np, axis=1)
            ax.hist(norms, bins=30, alpha=0.7, color='blue', density=True)
            ax.set_xlabel('Embedding Norm')
            ax.set_ylabel('Density')
            ax.set_title(f'Embedding Norm Distribution\n(Mean: {norms.mean():.3f}, Std: {norms.std():.3f})')
            ax.grid(True, alpha=0.3)
        except Exception as e:
            logger.warning(f"Norm distribution plot failed: {e}")
            ax.text(0.5, 0.5, 'Norm distribution\nplot failed', 
                   ha='center', va='center', transform=ax.transAxes)