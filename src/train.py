import os
import random

import hydra
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from models.osnet_ain import osnet_ain_x1_0, osnet_ain_x0_75, osnet_ain_x0_5, osnet_ain_x0_25


from datasets.generic_dataset import get_generic_datasets
from datasets.mnist_dataset import get_mnist_datasets
from datasets.reid_dataset import get_reid_datasets

from losses.contrastive_loss import ContrastiveLoss
from models.lightweight_embedder import LightweightEmbedder
from models.simple_cnn import SimpleCNN
from models.osnet_ain import OSNet

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Set device backend
        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            device = "cuda"
        # Check for MPS (Apple Silicon GPU)
        elif torch.backends.mps.is_available():
            device = "mps"
        # Fallback to CPU
        else:
            device = "cpu"
        
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        
        # Set random seeds
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        # Create save directory
        os.makedirs(cfg.trainer.save_dir, exist_ok=True)
        
        # Initialize MLflow
        self._init_mlflow()

        # Initialize components
        self.model = self._get_model()
        self.criterion = self._get_loss()
        self.optimizer = self._get_optimizer()
        
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self._get_dataset()

        # Training history - track every self.cfg.checkpoint_epoch batches
        self.train_losses = []
        self.val_losses = []
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_train_loss = float('inf')
        self.best_epoch = -1

    def _init_mlflow(self):
        """Initialize MLflow tracking"""
        # Set MLflow tracking URI (optional, defaults to local ./mlruns)
        if hasattr(self.cfg, 'mlflow') and hasattr(self.cfg.mlflow, 'tracking_uri'):
            print(f"Initializing mlflow")
            mlflow.set_tracking_uri(self.cfg.mlflow.tracking_uri)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Set experiment name
        experiment_name = getattr(self.cfg.mlflow, 'experiment_name', f'{self.cfg.model}_contrastive_reid_training') if hasattr(self.cfg, 'mlflow') else f'{self.cfg.model}_contrastive_reid_training'
        mlflow.set_experiment(experiment_name)
        
        # Set run name dynamically
        dynamic_run_name = (
            f"{self.cfg.model}_"
            f"lr-{self.cfg.trainer.learning_rate}_"
            f"bs-{self.cfg.dataset.batch_size}_"
            f"{timestamp}"  
        )
        
        # Start MLflow run
        mlflow.start_run(run_name=dynamic_run_name)
 
        # Log hyperparameters
        mlflow.log_params({
            "model_name": self.cfg.model.name,
            "model_variant": self.cfg.model.params.variant if self.cfg.model.name == "osnet_ain" else None,
            "embedding_dim": self.cfg.model.params.embedding_dim if hasattr(self.cfg.model.params, 'embedding_dim') else None,
            "learning_rate": self.cfg.trainer.learning_rate,
            "weight_decay": self.cfg.trainer.weight_decay,
            "max_epochs": self.cfg.trainer.max_epochs,
            "batch_size": self.cfg.dataset.batch_size,
            "loss_name": self.cfg.loss.name,
            "loss_margin": self.cfg.loss.margin,
            "distance_metric": self.cfg.loss.distance_metric,
            "dataset_name": self.cfg.dataset.name,
            "seed": self.cfg.seed,
            "device": str(self.device)
        })

    def _get_model(self):
        """Get model based on name and config"""
        model_name = self.cfg.model.name
        if model_name == "simple_cnn":
            model_config = self.cfg.model
            print(f"Creating SimpleCNN with params: {model_config.params}")
            return SimpleCNN(model_config.params.embedding_dim).to(self.device)
        elif model_name == "lightweight_embedder":
            model_config = self.cfg.model
            print(f"Creating LightweightEmbedder with params: {model_config.params}")
            return LightweightEmbedder(**model_config.params).to(self.device)
        elif model_name == "osnet_ain":
            model_config = self.cfg.model
            print(f"Creating OSNet with params: {model_config.params}")
            
            loss = 'triplet'       
                  
            # Force triplet loss for OSNet in Siamese training
            if hasattr(model_config.params, 'loss') and model_config.params.loss != 'triplet':
                print(f"Warning: OSNet loss was set to '{model_config.params.loss}', changing to 'triplet' for Siamese training")

                        
            # Import and instantiate the specific OSNet variant
            if model_config.params.variant == "x1_0":
                model = osnet_ain_x1_0(
                    num_classes=model_config.params.num_classes,
                    pretrained=model_config.params.pretrained,
                    loss=loss,  # Force triplet loss to get features
                    feature_dim=model_config.params.feature_dim
                )
            elif model_config.params.variant == "x0_75":
                model = osnet_ain_x0_75(
                    num_classes=model_config.params.num_classes,
                    pretrained=model_config.params.pretrained,
                    loss=loss, 
                    feature_dim=model_config.params.feature_dim
                )
            elif model_config.params.variant == "x0_5":
                model = osnet_ain_x0_5(
                    num_classes=model_config.params.num_classes,
                    pretrained=model_config.params.pretrained,
                    loss=loss, 
                    feature_dim=model_config.params.feature_dim
                )
            elif model_config.params.variant == "x0_25":
                model = osnet_ain_x0_25(
                    num_classes=model_config.params.num_classes,
                    pretrained=model_config.params.pretrained,
                    loss=loss, 
                    feature_dim=model_config.params.feature_dim
                )
                    
            # Ensure the model returns features, not classifications
            model.training = True  # Force training mode for feature extraction
                     
            return model.to(self.device)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def _get_loss(self):
        """Get loss function based on name and config"""
        if self.cfg.loss.name == "contrastive":
            return ContrastiveLoss(
                margin=self.cfg.loss.margin,
                distance_metric=self.cfg.loss.distance_metric,
            ).to(self.device)
        else:
            raise ValueError(f"Unknown loss: {self.cfg.loss.name}")

    def _get_optimizer(self):
        """Get optimizer"""
        return optim.Adam(
            self.model.parameters(),
            lr=self.cfg.trainer.learning_rate,
            weight_decay=self.cfg.trainer.weight_decay,
        )

    def _get_dataset(self):
        """Get dataset based on name and config"""
        dataset_name = self.cfg.dataset.name
        
        if dataset_name == "mnist_pairs":
            dataset_config = self.cfg.dataset
            return get_mnist_datasets(dataset_config)
        # TODO: user must implement this
        elif dataset_name == "generic_pairs":
            dataset_config = self.cfg.dataset
            return get_generic_datasets(dataset_config)
        elif dataset_name  == "reid_pairs":
            
            dataset_config = self.cfg.dataset
            return get_reid_datasets(dataset_config)
            
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_losses = []  # Track losses for saving
        batch_count = 0
        
        
        pbar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        for batch_idx, (img1, img2, labels) in enumerate(pbar):        
            img1, img2, labels = (
                img1.to(self.device),
                img2.to(self.device),
                labels.to(self.device),
            )

            self.optimizer.zero_grad()

            # Get embeddings for both images
            if hasattr(self.model, "forward_one"):
                # For siamese networks like SimpleCNN
                emb1, emb2 = self.model(img1, img2) 
            elif self.cfg.model.name == "osnet_ain":
                # OSNet returns (predictions, features) tuple in training mode with triplet loss
                output1 = self.model(img1)
                output2 = self.model(img2)
                
                # Handle tuple return from OSNet
                if isinstance(output1, tuple):
                    _, emb1 = output1  # Take features, ignore predictions
                    _, emb2 = output2
                else:
                    emb1, emb2 = output1, output2
        
            else:
                # For single-input networks like SimpleEmbedder, OsNet
                emb1 = self.model(img1)
                emb2 = self.model(img2)
                

            # Compute loss
            loss = self.criterion(emb1, emb2, labels.float())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())
            batch_count += 1

            if batch_idx % self.cfg.trainer.log_interval == 0:
                allocated = torch.mps.current_allocated_memory() / 1024**3
                print(f"MPS memory: {allocated:.2f} GB")
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Save train loss every checkpoint_epoch batches
            if batch_count % self.cfg.trainer.checkpoint_epoch == 0:
                loss_sum = sum(batch_losses[-self.cfg.trainer.checkpoint_epoch:])  
                loss_count = min(len(batch_losses), self.cfg.trainer.checkpoint_epoch)

                avg_loss = loss_sum / loss_count
                
                self.train_losses.append(avg_loss)
                # print(f"Average train loss over last {self.cfg.trainer.checkpoint_epoch} batches: {avg_loss:.4f}")
                
                # Log to MLflow
                mlflow.log_metric("train_loss_batches", avg_loss, step=len(self.train_losses))

        # Log epoch-level training loss
        epoch_avg_loss = total_loss / len(self.train_dataloader)
        mlflow.log_metric("train_loss_epoch", epoch_avg_loss, step=epoch)
        return epoch_avg_loss

    def validate_epoch(self, epoch):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_idx, (img1, img2, labels) in enumerate(
                tqdm(self.val_dataloader, desc="Validation")
            ):
                img1, img2, labels = (
                    img1.to(self.device),
                    img2.to(self.device),
                    labels.to(self.device),
                )

                # Get embeddings for both images
                if hasattr(self.model, "forward_one"):
                    # For siamese networks like SimpleCNN
                    emb1, emb2 = self.model(img1, img2)
                elif self.cfg.model.name == "osnet_ain":
                    # OsNet called for each image and may return a tuple.
                    output1 = self.model(img1)
                    output2 = self.model(img2)
                    if isinstance(output1, tuple):
                        _, emb1 = output1
                        _, emb2 = output2
                    else:
                        emb1, emb2 = output1, output2
                else:
                    # For single-input networks like SimpleEmbedder
                    emb1 = self.model(img1)
                    emb2 = self.model(img2)

                # Compute loss
                loss = self.criterion(emb1, emb2, labels.float())
                total_loss += loss.item()

        val_loss = total_loss / len(self.val_dataloader)
        # Log validation loss to MLflow
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        return val_loss


    def save_checkpoint(self, epoch, train_loss, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": self.cfg,
            "is_best": is_best,
            "best_val_loss": self.best_val_loss,
            "best_train_loss": self.best_train_loss,
            "best_epoch": self.best_epoch,
        }
        
        if is_best:
            # Save best model based on val loss
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            checkpoint_path = os.path.join(
                self.cfg.trainer.save_dir, f"{self.cfg.model.name}_best_{timestamp}_e{epoch + 1}.pth"
            )
            
            torch.save(checkpoint, checkpoint_path)
            mlflow.log_artifact(checkpoint_path) 
        
        
        else: 
            # Save model at set interval for recovery 
            checkpoint_path = os.path.join(
                self.cfg.trainer.save_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            )
            torch.save(checkpoint, checkpoint_path)
            mlflow.log_artifact(checkpoint_path)
            

    def save_final_model(self):
        """Save final model"""
        model_path = os.path.join(self.cfg.trainer.save_dir, "final_model.pth")
        torch.save(self.model.state_dict(), model_path)
    
        # Log final model to MLflow
        mlflow.log_artifact(model_path)
        mlflow.pytorch.log_model(self.model, "model")
 
 
    def plot_training_curves(self):
        """Plot training curves"""
        # Create figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot training loss (every epoch)
        if self.train_losses:
            ax1.plot(self.train_losses, label="Train Loss", color="blue") 
            
        ax1.set_xlabel(f"Batch (x{self.cfg.trainer.checkpoint_epoch})")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"Training Loss (every {self.cfg.trainer.checkpoint_epoch} batches)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot validation loss (every epoch)
        if self.val_losses:
            ax2.plot(self.val_losses, label="Val Loss", color="red")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation Loss (every epoch)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = f"{self.cfg.trainer.save_dir}/training_curves.png"
        plt.savefig(plot_path)

        # Log plot to MLflow
        mlflow.log_artifact(plot_path)

        plt.close()

    def train(self):
        """Main training loop"""
        print(f"Using device: {self.device}")
        print("Starting training...")

        try:
            epoch_pbar = tqdm(range(self.cfg.trainer.max_epochs), desc="Training")
            for epoch in epoch_pbar:
                epoch_pbar.set_description(f"Training: Epoch {epoch}")

                # Train
                train_loss = self.train_epoch(epoch)

                # Validate
                val_loss = self.validate_epoch(epoch)
                self.val_losses.append(val_loss)

                print(
                    f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Check if this epoch produced the best model (save every time we improve)
                if val_loss < self.best_val_loss:
                    print(f"New best model found at epoch {epoch + 1}! Val loss: {val_loss:.4f} (previous best: {self.best_val_loss:.4f})")

                    self.best_val_loss = val_loss
                    self.best_train_loss = train_loss
                    self.best_epoch = epoch
                    
                    # Save best model checkpoint
                    self.save_checkpoint(epoch, train_loss, val_loss, is_best=True)
                    
                    # Log best metrics to MLflow
                    mlflow.log_metric("best_val_loss", self.best_val_loss, step=epoch)
                    mlflow.log_metric("best_train_loss", self.best_train_loss, step=epoch)
                    
                    
                # Save regular checkpoint every checkpoint_epoch epochs (for recovery)
                if (epoch + 1) % self.cfg.trainer.checkpoint_epoch == 0:
                    print(f"Saving regular checkpoint at epoch {epoch + 1}")
                    self.save_checkpoint(epoch, train_loss, val_loss, is_best=False)

            # Save final model and plot
            self.save_final_model()
            self.plot_training_curves()
            print("Training completed!")
            
            print(f"Best model was found at epoch {self.best_epoch + 1}")
            print(f"Best validation loss: {self.best_val_loss:.4f}")
            print(f"Best training loss: {self.best_train_loss:.4f}")
                
        finally:
            # End MLflow run
            mlflow.end_run()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
