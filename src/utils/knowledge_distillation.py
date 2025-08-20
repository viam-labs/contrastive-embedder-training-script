from omegaconf import DictConfig
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import logging

# Import the loss from the losses folder
from losses.knowledge_distillation_loss import KnowledgeDistillationLoss

# Set up logger following your transfer_learning.py pattern
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('knowledge_distillation')
logger.setLevel(logging.INFO)


class KnowledgeDistillationManager:
    """
    Manager class for knowledge distillation operations.
    Follows the same pattern as your OSNetTransferLearning class.
    """
    
    def __init__(self):
        """Initialize the knowledge distillation manager"""
        self.model_variant_map = {
            'x1_0': {'channels': [64, 256, 384, 512], 'feature_dim': 512},
            'x0_75': {'channels': [48, 192, 288, 384], 'feature_dim': 384},
            'x0_5': {'channels': [32, 128, 192, 256], 'feature_dim': 256},
            'x0_25': {'channels': [16, 64, 96, 128], 'feature_dim': 128}
        }
    
    def load_teacher_model(self, checkpoint_path: str, model_variant: str, 
                          num_classes: int, device: torch.device) -> nn.Module:
        """
        Load teacher model from checkpoint
        
        Args:
            checkpoint_path: Path to teacher model checkpoint
            model_variant: Teacher model variant (x1_0, x0_75, etc.)
            num_classes: Number of classes
            device: Device to load model on
            
        Returns:
            Loaded teacher model
        """
        from models import osnet_ain_x1_0, osnet_ain_x0_75, osnet_ain_x0_5, osnet_ain_x0_25
        
        # Model variant mapping
        model_map = {
            'x1_0': osnet_ain_x1_0,
            'x0_75': osnet_ain_x0_75, 
            'x0_5': osnet_ain_x0_5,
            'x0_25': osnet_ain_x0_25
        }
        
        if model_variant not in model_map:
            raise ValueError(f"Unknown teacher model variant: {model_variant}")
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {checkpoint_path}")
        
        # Get model specifications
        model_specs = self.model_variant_map[model_variant]
        
        logger.info(f"Loading teacher model {model_variant} from: {checkpoint_path}")
        
        # Create teacher model
        model_fn = model_map[model_variant]
        teacher = model_fn(
            num_classes=num_classes,
            pretrained=False,
            loss='triplet',
            feature_dim=model_specs['feature_dim'],
            dropout_p=0.0  # No dropout during inference
        )
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats (same as your transfer learning code)
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Remove 'module.' prefix if present (from DataParallel)
        clean_state_dict = {}
        for key, value in state_dict.items():
            clean_key = key.replace('module.', '')
            clean_state_dict[clean_key] = value
        
        # Load state dict
        missing_keys, unexpected_keys = teacher.load_state_dict(clean_state_dict, strict=False)
        
        if missing_keys:
            logger.debug(f"Missing keys in teacher model: {missing_keys}")
        if unexpected_keys:
            logger.debug(f"Unexpected keys in checkpoint: {unexpected_keys}")
        
        teacher.eval()
        
        # Verify teacher model loaded correctly (same pattern as transfer learning)
        sample_weight = None
        for name, param in teacher.named_parameters():
            if 'conv1.conv.weight' in name:
                sample_weight = param.data.mean().item()
                break
        
        if sample_weight and abs(sample_weight) > 1e-6:
            logger.info(f"Teacher model {model_variant} loaded successfully (sample weight: {sample_weight:.6f})")
        else:
            logger.warning("Warning: Teacher model weights might be zero or very small")
        
        return teacher
    
    def wrap_loss_with_distillation(self, base_loss, cfg: DictConfig) -> KnowledgeDistillationLoss:
        """
        Wrap a base loss function with knowledge distillation
        
        Args:
            base_loss: Base loss function (ContrastiveLoss, TripletLoss, etc.)
            cfg: Configuration containing distillation parameters
            
        Returns:
            KnowledgeDistillationLoss wrapper
        """
        distillation_cfg = cfg.knowledge_distillation
        
        distillation_loss = KnowledgeDistillationLoss(
            base_loss=base_loss,
            temperature=distillation_cfg.get('temperature', 4.0),
            alpha=distillation_cfg.get('alpha', 0.7),
            feature_weight=distillation_cfg.get('feature_weight', 0.1),
            normalize_embeddings=cfg.loss.get('normalize_embeddings', True)
        )
        
        logger.info("Base loss wrapped with knowledge distillation")
        logger.info(f"  Temperature: {distillation_cfg.get('temperature', 4.0)}")
        logger.info(f"  Alpha: {distillation_cfg.get('alpha', 0.7)}")
        logger.info(f"  Feature weight: {distillation_cfg.get('feature_weight', 0.1)}")
        
        return distillation_loss
    
    def get_distillation_info_for_checkpoint(self, cfg: DictConfig) -> Dict[str, Any]:
        """
        Get knowledge distillation information for checkpoint saving
        
        Args:
            cfg: Configuration
            
        Returns:
            Dictionary with distillation info
        """
        if not hasattr(cfg, 'knowledge_distillation') or not cfg.knowledge_distillation.enabled:
            return {}
        
        distillation_cfg = cfg.knowledge_distillation
        
        return {
            'knowledge_distillation_enabled': True,
            'teacher_checkpoint': distillation_cfg.teacher_checkpoint,
            'teacher_variant': distillation_cfg.get('teacher_variant', 'x1_0'),
            'student_variant': cfg.model.params.variant,
            'temperature': distillation_cfg.get('temperature', 4.0),
            'alpha': distillation_cfg.get('alpha', 0.7),
            'feature_weight': distillation_cfg.get('feature_weight', 0.1)
        }


# Helper functions following your transfer_learning.py pattern
def apply_knowledge_distillation(base_loss, cfg: DictConfig, device: torch.device) -> Tuple[Optional[nn.Module], nn.Module]:
    """
    Apply knowledge distillation to enhance a base loss function.
    Follows the same pattern as apply_transfer_learning() in your codebase.
    
    Args:
        base_loss: Base loss function (ContrastiveLoss, TripletLoss, etc.)
        cfg: Configuration containing knowledge distillation settings
        device: Device to use
        
    Returns:
        Tuple of (teacher_model, enhanced_loss_function)
    """
    if not hasattr(cfg, 'knowledge_distillation') or not cfg.knowledge_distillation.enabled:
        return None, base_loss
    
    logger.info("="*60)
    logger.info("APPLYING KNOWLEDGE DISTILLATION")
    logger.info("="*60)
    
    # Initialize knowledge distillation manager
    kd_manager = KnowledgeDistillationManager()
    
    try:
        # Load teacher model
        teacher_model = kd_manager.load_teacher_model(
            checkpoint_path=cfg.knowledge_distillation.teacher_checkpoint,
            model_variant=cfg.knowledge_distillation.get('teacher_variant', 'x1_0'),
            num_classes=cfg.model.params.num_classes,
            device=device
        )
        
        # Wrap base loss with distillation
        distillation_loss = kd_manager.wrap_loss_with_distillation(base_loss, cfg)
        
        # Set teacher in the loss function
        distillation_loss.set_teacher_model(teacher_model, device)
        
        # Print summary (same style as your transfer learning)
        logger.info("="*60)
        logger.info("KNOWLEDGE DISTILLATION SUMMARY")
        logger.info("="*60)
        logger.info(f"Teacher model: {cfg.knowledge_distillation.get('teacher_variant', 'x1_0')}")
        logger.info(f"Student model: {cfg.model.params.variant}")
        logger.info(f"Base loss function: {cfg.loss.name}")
        logger.info(f"Temperature: {cfg.knowledge_distillation.get('temperature', 4.0)}")
        logger.info(f"Alpha (teacher weight): {cfg.knowledge_distillation.get('alpha', 0.7)}")
        logger.info(f"Feature matching weight: {cfg.knowledge_distillation.get('feature_weight', 0.1)}")
        logger.info("="*60)
        
        return teacher_model, distillation_loss
        
    except Exception as e:
        logger.error(f"Failed to setup knowledge distillation: {e}")
        raise


def add_distillation_info_to_checkpoint(checkpoint: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """
    Add knowledge distillation information to checkpoint if applicable.
    Follows the same pattern as add_transfer_info_to_checkpoint() in your codebase.
    
    Args:
        checkpoint: Checkpoint dictionary
        cfg: Configuration
        
    Returns:
        Updated checkpoint
    """
    kd_manager = KnowledgeDistillationManager()
    distillation_info = kd_manager.get_distillation_info_for_checkpoint(cfg)
    
    if distillation_info:
        checkpoint['knowledge_distillation_info'] = distillation_info
    
    return checkpoint