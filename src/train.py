import os
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from omegaconf import DictConfig
from tqdm import tqdm

from datasets.generic_dataset import get_generic_datasets
from datasets.mnist_dataset import get_mnist_datasets
from datasets.reid_dataset import get_reid_datasets

from losses.contrastive_loss import ContrastiveLoss
from models.lightweight_embedder import LightweightEmbedder
from models.simple_cnn import SimpleCNN

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        # Set device backend
        # Check for CUDA (NVIDIA GPU)∂
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

        # Initialize components
        self.model = self._get_model()
        self.criterion = self._get_loss()
        self.optimizer = self._get_optimizer()
        
        self.train_dataloader, self.val_dataloader, self.test_dataloader = self._get_dataset()

        # Training history - track every 10 batches
        self.train_losses = []
        self.val_losses = []
        self.batch_count = 0

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

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        batch_losses = []  # Track losses for saving
        batch_count = 0
        for batch_idx, (img1, img2, labels) in enumerate(
            tqdm(self.train_dataloader, desc="Training")
        ):
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
            else:
                # For single-input networks like SimpleEmbedder
                emb1 = self.model(img1)
                emb2 = self.model(img2)

            # Compute loss
            loss = self.criterion(emb1, emb2, labels.float())
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            batch_losses.append(loss.item())
            batch_count += 1
            print(batch_count)

            if batch_idx % self.cfg.trainer.log_interval == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")

            # Save train loss every 10 batches
            if batch_count % 10 == 0:
                avg_loss = sum(batch_losses[-10:]) / 10
                self.train_losses.append(avg_loss)
                print(f"Average train loss over last 10 batches: {avg_loss:.4f}")

        return total_loss / len(self.train_dataloader)

    def validate_epoch(self):
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
                else:
                    # For single-input networks like SimpleEmbedder
                    emb1 = self.model(img1)
                    emb2 = self.model(img2)

                # Compute loss
                loss = self.criterion(emb1, emb2, labels.float())
                total_loss += loss.item()

        return total_loss / len(self.val_dataloader)

    def save_checkpoint(self, epoch, train_loss, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "config": self.cfg,
        }
        torch.save(
            checkpoint,
            os.path.join(
                self.cfg.trainer.save_dir, f"checkpoint_epoch_{epoch + 1}.pth"
            ),
        )

    def save_final_model(self):
        """Save final model"""
        torch.save(
            self.model.state_dict(),
            os.path.join(self.cfg.trainer.save_dir, "final_model.pth"),
        )

    def plot_training_curves(self):
        """Plot training curves"""
        # Create figure with 2 subplots side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # Plot training loss (every 10 batches)
        ax1.plot(self.train_losses, label="Train Loss", color="blue")
        ax1.set_xlabel("Batch (x10)")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss (every 10 batches)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot validation loss (every epoch)
        ax2.plot(self.val_losses, label="Val Loss", color="red")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title("Validation Loss (every epoch)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("./training_curves.png")
        plt.close()

    def train(self):
        """Main training loop"""
        print(f"Using device: {self.device}")
        print("Starting training...")

        for epoch in range(self.cfg.trainer.max_epochs):
            print(f"\nEpoch {epoch + 1}/{self.cfg.trainer.max_epochs}")

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate_epoch()
            self.val_losses.append(val_loss)

            print(
                f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, train_loss, val_loss)

        # Save final model and plot
        self.save_final_model()
        self.plot_training_curves()
        print("Training completed!")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
