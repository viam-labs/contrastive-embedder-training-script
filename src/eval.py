#!/usr/bin/env python3
"""
Enhanced script to evaluate a locally saved checkpoint using Hydra configs
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
from datetime import datetime

# Add src directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src') if 'src' not in current_dir else current_dir
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Also add parent directory if we're in src
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import your existing components
try:
    from models.osnet_ain import osnet_ain_x1_0, osnet_ain_x0_75, osnet_ain_x0_5, osnet_ain_x0_25
    from datasets.generic_dataset import get_generic_datasets
    from datasets.mnist_dataset import get_mnist_datasets
    from datasets.reid_dataset import get_reid_datasets
    from losses.contrastive_loss import ContrastiveLoss
    from models.lightweight_embedder import LightweightEmbedder
    from models.simple_cnn import SimpleCNN
except ImportError:
    # Try with src prefix
    from src.models.osnet_ain import osnet_ain_x1_0, osnet_ain_x0_75, osnet_ain_x0_5, osnet_ain_x0_25
    from src.datasets.generic_dataset import get_generic_datasets
    from src.datasets.mnist_dataset import get_mnist_dataset
    from src.datasets.reid_dataset import get_reid_datasets
    from src.losses.contrastive_loss import ContrastiveLoss
    from src.models.lightweight_embedder import LightweightEmbedder
    from src.models.simple_cnn import SimpleCNN


class HydraCheckpointEvaluator:
    def __init__(self, checkpoint_path: str, cfg: DictConfig, test_dataloader=None):
        """
        Initialize evaluator with local checkpoint and Hydra config
        
        Args:
            checkpoint_path: Path to your checkpoint file
            cfg: Hydra config (fully composed)
            test_dataloader: Optional pre-existing test dataloader
        """
        self.checkpoint_path = checkpoint_path
        self.cfg = cfg
        
        # Set device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        self.checkpoint = self._load_checkpoint()
        
        # Load checkpoint config
                # Get the config from the checkpoint if available, otherwise use the Hydra config
        checkpoint_cfg = self.checkpoint.get('config')
        if checkpoint_cfg:
            print("Using model configuration from checkpoint.")
            # Use OmegaConf.create to convert the dict into a DictConfig
            # This is important for Hydra's dot notation to work
            self.model_cfg = OmegaConf.create(checkpoint_cfg)
        else:
            print("Using model configuration from Hydra config.")
            self.model_cfg = self.cfg
        
        # Pass the correct config to the model creation method
        self.model = self._get_model_from_config(self.model_cfg)
        print(f"Model created: {self.model_cfg.model.name} with variant {self.model_cfg.model.params.get('variant', 'default')}")
        
        # Load model weights
        self._load_model_weights()
        
        # Initialize loss and dataset
        self.criterion = self._get_loss()
        
        # Use provided test_dataloader or create one
        if test_dataloader is not None:
            print("Using provided test_dataloader")
            self.test_dataloader = test_dataloader
        else:
            print("Creating test_dataloader from config")
            _, _, self.test_dataloader = self._get_dataset()

    def _load_checkpoint(self):
        """Load checkpoint"""
        print(f"Loading checkpoint from: {self.checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        print(f"Checkpoint keys: {list(checkpoint.keys())}")
        
        if 'config' in checkpoint:
            print("Note: Checkpoint contains config but using Hydra config instead")
        
        return checkpoint

        
    def _get_model_from_config(self, cfg):
        """Recreate model from config"""
        model_name = cfg.model.name
        
        print(f"Creating model: {model_name}")
        
        if model_name == "simple_cnn":
            embedding_dim = cfg.model.params.get('embedding_dim', 128)
            return SimpleCNN(embedding_dim).to(self.device)
            
        elif model_name == "lightweight_embedder":
            return LightweightEmbedder(**cfg.model.params).to(self.device)
            
        elif model_name == "osnet_ain":
            loss = 'triplet'
            
            variant_map = {
                "x1_0": osnet_ain_x1_0,
                "x0_75": osnet_ain_x0_75, 
                "x0_5": osnet_ain_x0_5,
                "x0_25": osnet_ain_x0_25
            }
            
            variant = cfg.model.params.get('variant', 'x1_0')
            if variant in variant_map:
                print(f"Creating OSNet-AIN variant: {variant}")
                model = variant_map[variant](
                    num_classes=cfg.model.params.get('num_classes', 1000),
                    pretrained=cfg.model.params.get('pretrained', False),
                    loss=loss,
                    feature_dim=cfg.model.params.get('feature_dim', 512)
                )
            else:
                raise ValueError(f"Unknown variant: {variant}")
            
            return model.to(self.device)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # In HydraCheckpointEvaluator class
    def _load_model_weights(self):
        """Load model weights from checkpoint"""
        
        # Try to find the state dict in one of three ways
        if 'state_dict' in self.checkpoint:
            state_dict = self.checkpoint['state_dict']
        elif 'model_state_dict' in self.checkpoint:
            state_dict = self.checkpoint['model_state_dict']
        # NEW LOGIC: Check if the state dict is the top-level dictionary itself
        elif 'conv1.conv.weight' in self.checkpoint:
            state_dict = self.checkpoint
        else:
            raise ValueError("No state_dict found in checkpoint under 'state_dict', 'model_state_dict', or as the top-level keys.")
        
        print(f"Loaded state_dict with {len(state_dict)} parameters.")
        
        # The rest of this method remains unchanged
        # Try to load state dict, handling potential key mismatches
        try:
            self.model.load_state_dict(state_dict, strict=True)
            print("Model loaded successfully (strict)")
        except RuntimeError as e:
            print(f"Strict loading failed: {e}")
            print("Trying non-strict loading...")
            
            # Non-strict loading
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}")
            
            print("Model loaded successfully (non-strict)")
        
        self.model.eval()

    def _get_loss(self):
        """Get loss function from config"""
        if self.cfg.loss.name == "contrastive":
            return ContrastiveLoss(
                margin=self.cfg.loss.get('margin', 1.0),
                distance_metric=self.cfg.loss.get('distance_metric', 'euclidean'),
            ).to(self.device)
        else:
            raise ValueError(f"Unknown loss: {self.cfg.loss.name}")

    def _get_dataset(self):
        """Get dataset from config"""
        dataset_name = self.cfg.dataset.name
        
        print(f"Loading dataset: {dataset_name}")
        
        if dataset_name == "mnist_pairs":
            return get_mnist_datasets(self.cfg.dataset)
        elif dataset_name == "generic_pairs":
            return get_generic_datasets(self.cfg.dataset)
        elif dataset_name == "reid_pairs":
            return get_reid_datasets(self.cfg.dataset)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def set_test_dataloader(self, test_dataloader):
        """Set a custom test dataloader after initialization"""
        print("Setting custom test_dataloader")
        self.test_dataloader = test_dataloader

    def evaluate(self, save_plots=True):
        """Evaluate the model and return metrics"""
        print("Starting evaluation...")
        self.model.eval()
        
        all_distances = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (img1, img2, labels) in enumerate(
                tqdm(self.test_dataloader, desc="Evaluation")
            ):
                img1, img2, labels = (
                    img1.to(self.device),
                    img2.to(self.device),
                    labels.to(self.device),
                )

                # Get embeddings
                if hasattr(self.model, "forward_one"):
                    emb1, emb2 = self.model(img1, img2)
                elif self.cfg.model.name == "osnet_ain":
                    output1 = self.model(img1)
                    output2 = self.model(img2)
                    
                    if isinstance(output1, tuple):
                        _, emb1 = output1
                        _, emb2 = output2
                    else:
                        emb1, emb2 = output1, output2
                else:
                    emb1 = self.model(img1)
                    emb2 = self.model(img2)

                # Compute distances
                if hasattr(self.criterion, '_compute_distance'):
                    distances = self.criterion._compute_distance(emb1, emb2)
                else:
                    # Fallback distance computation
                    if self.cfg.loss.distance_metric == 'cosine':
                        emb1_norm = torch.nn.functional.normalize(emb1, p=2, dim=1)
                        emb2_norm = torch.nn.functional.normalize(emb2, p=2, dim=1)
                        cosine_sim = torch.sum(emb1_norm * emb2_norm, dim=1)
                        distances = 1 - cosine_sim
                    else:
                        distances = torch.norm(emb1 - emb2, p=2, dim=1)
                
                all_distances.extend(distances.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Convert to numpy
        all_distances = np.array(all_distances)
        all_labels = np.array(all_labels)
        
        # Compute metrics
        metrics = self._compute_metrics(all_distances, all_labels)
        
        if save_plots:
            self._save_plots(all_distances, all_labels, metrics)
        
        return metrics

    def _compute_metrics(self, distances, labels):
        """Compute evaluation metrics including ReID-specific metrics"""
        scores = -distances
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = -thresholds[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        
        predictions = (distances <= optimal_threshold).astype(int)
        accuracy = np.mean(predictions == labels)
        
        # ReID-specific metrics
        # True Positives: correctly identified as same person
        # False Positives: incorrectly identified as same person  
        # True Negatives: correctly identified as different person
        # False Negatives: incorrectly identified as different person
        
        tp = np.sum((predictions == 1) & (labels == 1))
        fp = np.sum((predictions == 1) & (labels == 0)) 
        tn = np.sum((predictions == 0) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        
        # Precision: Of all pairs we said were same person, how many actually were?
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall (Sensitivity/TPR): Of all actual same person pairs, how many did we find?
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Specificity (TNR): Of all actual different person pairs, how many did we correctly reject?
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # F1 Score: Harmonic mean of precision and recall
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # False Acceptance Rate (FAR): How often do we incorrectly accept different people as same?
        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Rejection Rate (FRR): How often do we incorrectly reject same people as different?
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return {
            'auc': roc_auc,
            'accuracy': accuracy,
            'optimal_threshold': optimal_threshold,
            'optimal_tpr': optimal_tpr,
            'optimal_fpr': optimal_fpr,
            
            # ReID-specific metrics
            'precision': precision,
            'recall': recall,
            'specificity': specificity, 
            'f1_score': f1_score,
            'far': far,  # False Acceptance Rate
            'frr': frr,  # False Rejection Rate
            
            # Confusion matrix
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            
            # Distance statistics
            'mean_distance_same': np.mean(distances[labels == 1]),
            'mean_distance_diff': np.mean(distances[labels == 0]),
            'std_distance_same': np.std(distances[labels == 1]),
            'std_distance_diff': np.std(distances[labels == 0])
        }

    def _save_plots(self, distances, labels, metrics):
        """Save evaluation plots"""
        scores = -distances
        fpr, tpr, _ = roc_curve(labels, scores)
        
        plt.figure(figsize=(15, 5))
        
        # ROC curve
        plt.subplot(1, 3, 1)
        plt.plot(fpr, tpr, color="darkorange", lw=2, 
                label=f"ROC curve (AUC = {metrics['auc']:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
        plt.scatter(metrics['optimal_fpr'], metrics['optimal_tpr'], 
                   color='red', s=100, label=f"Optimal", zorder=5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Distance distributions
        plt.subplot(1, 3, 2)
        same_distances = distances[labels == 1]
        diff_distances = distances[labels == 0]
        
        plt.hist(same_distances, bins=50, alpha=0.7, label='Same person', color='green', density=True)
        plt.hist(diff_distances, bins=50, alpha=0.7, label='Different person', color='red', density=True)
        plt.axvline(metrics['optimal_threshold'], color='black', linestyle='--', 
                   label=f"Threshold = {metrics['optimal_threshold']:.3f}")
        plt.xlabel("Distance")
        plt.ylabel("Density")
        plt.title("Distance Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(1, 3, 3)
        data_to_plot = [same_distances, diff_distances]
        box_plot = plt.boxplot(data_to_plot, labels=['Same', 'Different'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightgreen')
        box_plot['boxes'][1].set_facecolor('lightcoral')
        plt.ylabel("Distance")
        plt.title("Distance Box Plot")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Create filename with model info
        model_info = f"{self.cfg.model.name}"
        if hasattr(self.cfg.model.params, 'variant'):
            model_info += f"_{self.cfg.model.params.variant}"
        
        plot_path = f"evaluation_{model_info}_{self.cfg.dataset.name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Evaluation plots saved to: {plot_path}")

    def print_results(self, metrics):
        """Print evaluation results"""
        print("\n" + "="*70)
        print("REID CHECKPOINT EVALUATION RESULTS")
        print("="*70)
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Model: {self.cfg.model.name}")
        if hasattr(self.cfg.model.params, 'variant'):
            print(f"Variant: {self.cfg.model.params.variant}")
        print(f"Epoch: {self.checkpoint.get('epoch', 'Unknown')}")
        print(f"Dataset: {self.cfg.dataset.name}")
        print("-"*70)
        
        # Primary ReID metrics
        print("ReID PERFORMANCE METRICS:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"Specificity: {metrics['specificity']:.4f}")
        print("-"*70)
        
        # Error rates (important for ReID systems)
        print("ERROR RATES:")
        print(f"False Acceptance Rate (FAR): {metrics['far']:.4f}")
        print(f"False Rejection Rate (FRR): {metrics['frr']:.4f}")
        print("-"*70)
        
        # Traditional classification metrics
        print("CLASSIFICATION METRICS:")
        print(f"AUC Score: {metrics['auc']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"TPR at Optimal: {metrics['optimal_tpr']:.4f}")
        print(f"FPR at Optimal: {metrics['optimal_fpr']:.4f}")
        print("-"*70)
        
        # Confusion matrix
        print("CONFUSION MATRIX:")
        print(f"True Positives (Same person correctly identified): {metrics['true_positives']}")
        print(f"False Positives (Different person incorrectly matched): {metrics['false_positives']}")
        print(f"True Negatives (Different person correctly rejected): {metrics['true_negatives']}")
        print(f"False Negatives (Same person incorrectly rejected): {metrics['false_negatives']}")
        print("-"*70)
        
        # Distance analysis
        print("DISTANCE ANALYSIS:")
        print(f"Optimal Threshold: {metrics['optimal_threshold']:.4f}")
        print(f"Mean Distance (Same): {metrics['mean_distance_same']:.4f} ± {metrics['std_distance_same']:.4f}")
        print(f"Mean Distance (Diff): {metrics['mean_distance_diff']:.4f} ± {metrics['std_distance_diff']:.4f}")
        print("="*70)
        
        # Performance interpretation
        print("\nPERFORMANCE INTERPRETATION:")
        if metrics['precision'] > 0.95 and metrics['recall'] > 0.95:
            print("✓ Excellent ReID performance")
        elif metrics['precision'] > 0.90 and metrics['recall'] > 0.90:
            print("✓ Good ReID performance") 
        elif metrics['precision'] > 0.80 and metrics['recall'] > 0.80:
            print("⚠ Moderate ReID performance")
        else:
            print("⚠ Poor ReID performance - consider model improvements")
            
        if metrics['far'] < 0.01:
            print("✓ Very low false acceptance rate")
        elif metrics['far'] < 0.05:
            print("✓ Acceptable false acceptance rate")
        else:
            print("⚠ High false acceptance rate - may cause incorrect matches")
            
        if metrics['frr'] < 0.05:
            print("✓ Very low false rejection rate")
        elif metrics['frr'] < 0.10:
            print("✓ Acceptable false rejection rate")
        else:
            print("⚠ High false rejection rate - may miss valid matches")

# The corrected function definition, with 'self' removed
def _write_results_to_file(file_path, checkpoint_path, metrics):
    """Writes formatted evaluation results to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output = f"""
    ============================================================
    EVALUATION RESULTS FOR CHECKPOINT: {os.path.basename(checkpoint_path)}
    Timestamp: {timestamp}
    ------------------------------------------------------------
    REID PERFORMANCE METRICS:
    Precision: {metrics['precision']:.4f}
    Recall: {metrics['recall']:.4f}
    F1-Score: {metrics['f1_score']:.4f}
    Specificity: {metrics['specificity']:.4f}

    ERROR RATES:
    False Acceptance Rate (FAR): {metrics['far']:.4f}
    False Rejection Rate (FRR): {metrics['frr']:.4f}

    CLASSIFICATION METRICS:
    AUC Score: {metrics['auc']:.4f}
    Accuracy: {metrics['accuracy']:.4f}
    Optimal Threshold: {metrics['optimal_threshold']:.4f}
    ============================================================
    """
    with open(file_path, 'a') as f:
        f.write(output)
    print(f"\nResults written to: {file_path}")
    

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main function using Hydra for configuration"""

    if not hasattr(cfg, 'checkpoint_path') or cfg.checkpoint_path is None:
        print("Usage: python src/eval.py checkpoint_path=<path_to_checkpoint>")
        return

    checkpoint_path = cfg.checkpoint_path
    
    # [New logic to handle results file]
    results_dir = "evaluation_logs"
    os.makedirs(results_dir, exist_ok=True)
    results_file_path = os.path.join(results_dir, "evaluation_results.txt")
    
    already_evaluated = False
    if os.path.exists(results_file_path):
        with open(results_file_path, 'r') as f:
            if os.path.basename(checkpoint_path) in f.read():
                already_evaluated = True
    
    if already_evaluated:
        print(f"Skipping checkpoint '{os.path.basename(checkpoint_path)}' as it has already been evaluated.")
        print(f"Results are in {results_file_path}")
        return
    
    # Rest of the evaluation logic
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        return
    
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Model: {cfg.model.name}")
    print(f"Dataset: {cfg.dataset.name}")
    
    try:
        evaluator = HydraCheckpointEvaluator(checkpoint_path, cfg)
        metrics = evaluator.evaluate(save_plots=True)
        evaluator.print_results(metrics)
        _write_results_to_file(results_file_path, checkpoint_path, metrics)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()