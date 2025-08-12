from omegaconf import DictConfig
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple, Any
import torch.nn.utils.prune as prune
import logging

# Set up a logger instance for the module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('osnet_transfer')
logger.setLevel(logging.ERROR)


class OSNetTransferLearning:
    """
    Utilities for transferring knowledge from a large OSNet model to a smaller one.
    """
    
    def __init__(self):
        """Initializes the transfer learning helper."""
        self.channel_mapping = {
            'x1_0': [64, 256, 384, 512],
            'x0_75': [48, 192, 288, 384],
            'x0_5': [32, 128, 192, 256],
            'x0_25': [16, 64, 96, 128]
        }
    
    def load_source_weights(self, checkpoint_path: str) -> Dict:
        """Loads weights from a source model checkpoint."""
        logger.info(f"Loading source weights from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)
        logger.info("Keys in the source state_dict:")
        for key in state_dict.keys():
            logger.info(key)
        
        return checkpoint.get('state_dict', checkpoint)
    
    def _transfer_conv_weights(self, source_weight: torch.Tensor, target_shape: Tuple, method: str) -> torch.Tensor:
        """Transfers convolutional weights by importance, center, or random selection."""
        src_out, src_in = source_weight.shape[:2]
        tgt_out, tgt_in = target_shape[:2]
        target_weight = torch.zeros(target_shape)
        
        if method == 'importance':
            out_importance = source_weight.norm(p=2, dim=(1, 2, 3))
            out_indices = out_importance.argsort(descending=True)[:tgt_out]
            in_importance = source_weight.norm(p=2, dim=(0, 2, 3))  
            in_indices = in_importance.argsort(descending=True)[:tgt_in]
            target_weight[:, :] = source_weight[out_indices][:, in_indices]
            logger.debug(f"Transferred conv weights using 'importance' method. Original shape: {source_weight.shape}, Target shape: {target_shape}")
        elif method == 'center':
            out_start, in_start = (src_out - tgt_out) // 2, (src_in - tgt_in) // 2
            target_weight = source_weight[out_start:out_start + tgt_out, in_start:in_start + tgt_in]
            logger.debug(f"Transferred conv weights using 'center' method. Original shape: {source_weight.shape}, Target shape: {target_shape}")
        elif method == 'random':
            out_indices, in_indices = torch.randperm(src_out)[:tgt_out], torch.randperm(src_in)[:tgt_in]
            target_weight[:, :] = source_weight[out_indices][:, in_indices]
            logger.debug(f"Transferred conv weights using 'random' method. Original shape: {source_weight.shape}, Target shape: {target_shape}")
        
        return target_weight
    
    def _transfer_bn_weights(self, source_params: Dict[str, torch.Tensor], target_channels: int, method: str) -> Dict[str, torch.Tensor]:
        """Transfers batch normalization parameters based on importance, center, or random selection."""
        source_channels = source_params['weight'].shape[0]
        if method == 'importance':
            indices = source_params['weight'].abs().argsort(descending=True)[:target_channels]
            logger.debug(f"Transferred BN weights using 'importance' method. Original channels: {source_channels}, Target channels: {target_channels}")
        elif method == 'center':
            start = (source_channels - target_channels) // 2
            indices = torch.arange(start, start + target_channels)
            logger.debug(f"Transferred BN weights using 'center' method. Original channels: {source_channels}, Target channels: {target_channels}")
        else:
            indices = torch.randperm(source_channels)[:target_channels]
            logger.debug(f"Transferred BN weights using 'random' method. Original channels: {source_channels}, Target channels: {target_channels}")
        
        return {
            'weight': source_params['weight'][indices],
            'bias': source_params['bias'][indices],
            'running_mean': source_params['running_mean'][indices],
            'running_var': source_params['running_var'][indices]
        }
    
    def _transfer_linear_weights(self, source_weight: torch.Tensor, source_bias: Optional[torch.Tensor], target_shape: Tuple, method: str) -> Tuple:
        """Transfers linear layer weights based on importance, center, or random selection."""
        src_out, src_in = source_weight.shape
        tgt_out, tgt_in = target_shape
        
        if method == 'importance':
            out_importance = source_weight.norm(dim=1)
            out_indices = out_importance.argsort(descending=True)[:tgt_out]
            in_importance = source_weight.norm(dim=0)
            in_indices = in_importance.argsort(descending=True)[:tgt_in]
            target_weight = source_weight[out_indices][:, in_indices]
            target_bias = source_bias[out_indices] if source_bias is not None else None
            logger.debug(f"Transferred linear weights using 'importance' method. Original shape: {source_weight.shape}, Target shape: {target_shape}")
        elif method == 'center':
            out_start, in_start = (src_out - tgt_out) // 2, (src_in - tgt_in) // 2
            target_weight = source_weight[out_start:out_start + tgt_out, in_start:in_start + tgt_in]
            target_bias = source_bias[out_start:out_start + tgt_out] if source_bias is not None else None
            logger.debug(f"Transferred linear weights using 'center' method. Original shape: {source_weight.shape}, Target shape: {target_shape}")
        else:
            out_indices, in_indices = torch.randperm(src_out)[:tgt_out], torch.randperm(src_in)[:tgt_in]
            target_weight = source_weight[out_indices][:, in_indices]
            target_bias = source_bias[out_indices] if source_bias is not None else None
            logger.debug(f"Transferred linear weights using 'random' method. Original shape: {source_weight.shape}, Target shape: {target_shape}")
        
        return target_weight, target_bias
    
    def transfer_osnet_weights(self, source_state_dict: Dict, target_model: nn.Module,
                              source_variant: str = 'x1_0', target_variant: str = 'x0_5',
                              method: str = 'importance', verbose: bool = True) -> nn.Module:
        """Transfers weights from a source OSNet model to a target OSNet model."""
        target_state_dict = target_model.state_dict()
        transferred_layers, skipped_layers = [], []
        
        logger.info(f"Starting weight transfer from {source_variant} to {target_variant} using '{method}' method.")
        
        for name, target_param in target_state_dict.items():
            if name not in source_state_dict or source_state_dict[name].shape == target_param.shape:
                if name in source_state_dict:
                    target_state_dict[name] = source_state_dict[name]
                    transferred_layers.append(f"{name} (exact match)")
                    logger.debug(f"Transferring {name} (exact match).")
                else:
                    skipped_layers.append(f"{name} (not in source)")
                    logger.warning(f"Skipping {name}: not found in source state_dict.")
                continue

            source_param = source_state_dict[name]

            if 'conv' in name and len(source_param.shape) == 4:
                target_state_dict[name] = self._transfer_conv_weights(source_param, target_param.shape, method)
                transferred_layers.append(f"{name} (conv: {source_param.shape} -> {target_param.shape})")
                logger.debug(f"Transferred {name} (conv) from {source_param.shape} to {target_param.shape}.")
            elif 'bn' in name and 'weight' in name:
                base_name = name.replace('.weight', '')
                
                # Use a dictionary to store found parameters
                source_params = {}
                for p in ['weight', 'bias', 'running_mean', 'running_var']:
                    source_key = f"{base_name}.{p}"
                    if source_key in source_state_dict:
                        source_params[p] = source_state_dict[source_key]
                    else:
                        logger.warning(f"Skipping BN parameter '{source_key}': not found in source state_dict.")
                
                # Proceed with transfer only if enough parameters were found
                if len(source_params) == 4:
                    transferred_bn_params = self._transfer_bn_weights(source_params, target_param.shape[0], method)
                    for p_name, p_val in transferred_bn_params.items():
                        target_state_dict[f"{base_name}.{p_name}"] = p_val
                    transferred_layers.append(f"{base_name} (BN: {source_params['weight'].shape[0]} -> {target_param.shape[0]})")
                else:
                    skipped_layers.append(f"{base_name} (BN missing params)")
            elif ('fc' in name or 'classifier' in name) and 'weight' in name:
                base_name, bias_name = name.replace('.weight', ''), f"{name.replace('.weight', '')}.bias"
                source_bias = source_state_dict.get(bias_name, None)
                w, b = self._transfer_linear_weights(source_param, source_bias, target_param.shape, method)
                target_state_dict[name] = w
                if b is not None and bias_name in target_state_dict:
                    target_state_dict[bias_name] = b
                transferred_layers.append(f"{name} (FC: {source_param.shape} -> {target_param.shape})")
                logger.debug(f"Transferred {name} (FC) from {source_param.shape} to {target_param.shape}.")
            else:
                skipped_layers.append(f"{name} (incompatible shape)")
                logger.warning(f"Skipping {name}: incompatible shape {source_param.shape} vs {target_param.shape}.")
        
        target_model.load_state_dict(target_state_dict, strict=False)
        logger.info(f"Loaded transferred state dictionary into the target model.")
        
        if verbose:
            logger.info(f"\n=== Transfer Learning Summary ===")
            logger.info(f"Source variant: {source_variant}")
            logger.info(f"Target variant: {target_variant}")
            logger.info(f"Transfer method: {method}")
            logger.info(f"Transferred layers: {len(transferred_layers)}")
            logger.info(f"Skipped layers: {len(skipped_layers)}")
            for layer in transferred_layers:
                logger.debug(f"Transferred: {layer}")
            for layer in skipped_layers:
                logger.debug(f"Skipped: {layer}")
        return target_model
    
    def apply_structured_pruning(self, model: nn.Module, pruning_ratio: float = 0.3, importance_type: str = 'l2') -> nn.Module:
        """Applies structured pruning to reduce model size."""
        logger.info(f"Applying structured pruning with a ratio of {pruning_ratio} using '{importance_type}' importance.")
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                if importance_type == 'l2': importance = module.weight.data.norm(2, dim=(1, 2, 3))
                elif importance_type == 'l1': importance = module.weight.data.norm(1, dim=(1, 2, 3))
                else: 
                    logger.warning(f"Unsupported importance type '{importance_type}'. Skipping pruning for {name}.")
                    continue
                num_prune = int(importance.shape[0] * pruning_ratio)
                if num_prune > 0:
                    prune.ln_structured(module, name='weight', amount=pruning_ratio, n=2, dim=0)
                    logger.debug(f"Pruned {num_prune} channels from {name}.")
        logger.info("Structured pruning complete.")
        return model


# Helper functions for transfer learning
def transfer_x1_0_to_x0_5(source_checkpoint_path: str, target_model, method: str = 'importance',
                          apply_pruning: bool = False, pruning_ratio: float = 0.2):
    """Convenient function to transfer from x1_0 to x0_5."""
    helper = OSNetTransferLearning()
    source_weights = helper.load_source_weights(source_checkpoint_path)
    target_model = helper.transfer_osnet_weights(source_weights, target_model, source_variant='x1_0',
                                                  target_variant='x0_5', method=method, verbose=True)
    if apply_pruning:
        target_model = helper.apply_structured_pruning(target_model, pruning_ratio=pruning_ratio)
    return target_model


def apply_transfer_learning(model: torch.nn.Module, cfg: DictConfig, device: torch.device) -> torch.nn.Module:
    """
    Apply transfer learning to a model based on configuration.
    
    Args:
        model: Target model to transfer weights to
        cfg: Configuration containing transfer learning settings
        device: Device to use for model
        
    Returns:
        Model with transferred weights
    """
    if not hasattr(cfg, 'transfer_learning') or not cfg.transfer_learning.enabled:
        return model
    
    # Suppress verbose warnings from transfer learning
    logging.getLogger('osnet_transfer').setLevel(logging.ERROR)
    
    logger.info("="*60)
    logger.info("APPLYING TRANSFER LEARNING")
    logger.info("="*60)
    
    # Initialize transfer learning helper
    tl_helper = OSNetTransferLearning()
    
    # Load source weights
    try:
        source_checkpoint = cfg.transfer_learning.source_checkpoint
        if not Path(source_checkpoint).exists():
            raise FileNotFoundError(f"Source checkpoint not found: {source_checkpoint}")
            
        source_weights = tl_helper.load_source_weights(source_checkpoint)
        logger.info(f"Loaded source checkpoint: {source_checkpoint}")
    except Exception as e:
        logger.error(f"Failed to load source weights: {e}")
        raise
    
    # Count parameters before transfer
    params_before = sum(p.numel() for p in model.parameters())
    
    # Transfer weights
    model = tl_helper.transfer_osnet_weights(
        source_weights,
        model,
        source_variant=cfg.transfer_learning.source_variant,
        target_variant=cfg.model.params.variant,
        method=cfg.transfer_learning.method,
        verbose=False  # Keep output clean
    )
    
    # Verify transfer by checking a sample weight
    sample_weight = None
    for name, param in model.named_parameters():
        if 'conv1.conv.weight' in name:
            sample_weight = param.data.mean().item()
            break
    
    # Print summary
    params_after = sum(p.numel() for p in model.parameters())
    logger.info("="*60)
    logger.info("TRANSFER LEARNING SUMMARY")
    logger.info("="*60)
    logger.info(f"Source model: {cfg.transfer_learning.source_variant}")
    logger.info(f"Target model: {cfg.model.params.variant}")
    logger.info(f"Transfer method: {cfg.transfer_learning.method}")
    logger.info(f"Model parameters: {params_after:,}")
    
    if sample_weight and abs(sample_weight) > 1e-6:
        logger.info(f"Weights successfully transferred (sample mean: {sample_weight:.6f})")
    else:
        logger.warning("Warning: Transferred weights might be zero or very small")
    
    # Apply optional pruning
    if hasattr(cfg.transfer_learning, 'pruning') and cfg.transfer_learning.pruning.enabled:
        logger.info(f"Applying pruning (ratio: {cfg.transfer_learning.pruning.ratio})")
        model = tl_helper.apply_structured_pruning(
            model,
            pruning_ratio=cfg.transfer_learning.pruning.ratio,
            importance_type=cfg.transfer_learning.pruning.get('importance_type', 'l2')
        )
        pruned_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Parameters after pruning: {pruned_params:,} ({(1-pruned_params/params_after)*100:.1f}% reduction)")
    
    logger.info("="*60)
    
    return model


def freeze_layers(model: torch.nn.Module, cfg: DictConfig) -> Dict[str, int]:
    """
    Freeze early layers of the model for transfer learning.
    
    Args:
        model: Model to freeze layers in
        cfg: Configuration with freeze settings
        
    Returns:
        Dictionary with freeze statistics
    """
    if not hasattr(cfg, 'transfer_learning') or not cfg.transfer_learning.enabled:
        return {'frozen': 0, 'total': 0}
    
    if not hasattr(cfg.transfer_learning, 'freeze_layers') or cfg.transfer_learning.freeze_layers <= 0:
        return {'frozen': 0, 'total': 0}
    
    num_layer_groups = cfg.transfer_learning.freeze_layers
    layer_groups = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']
    frozen_groups = layer_groups[:num_layer_groups]
    
    frozen_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_params += 1
        if any(group in name for group in frozen_groups):
            param.requires_grad = False
            frozen_params += 1
    
    logger.info(f"Frozen {frozen_params}/{total_params} parameters from layers: {frozen_groups}")
    
    return {'frozen': frozen_params, 'total': total_params, 'groups': frozen_groups}


def get_optimizer_with_differential_lr(model: torch.nn.Module, cfg: DictConfig):
    """
    Create optimizer with differential learning rates for transfer learning.
    
    Args:
        model: Model to optimize
        cfg: Configuration with optimizer settings
        
    Returns:
        Configured optimizer
    """
    # Check if differential learning rates are enabled
    use_differential = (
        hasattr(cfg, 'transfer_learning') and 
        cfg.transfer_learning.enabled and 
        hasattr(cfg.transfer_learning, 'differential_lr') and
        cfg.transfer_learning.differential_lr.enabled
    )
    
    if use_differential:
        # Separate parameters into backbone and head
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # Skip frozen parameters
                
            if 'fc' in name or 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
        
        # Get learning rate scale
        lr_scale = cfg.transfer_learning.differential_lr.backbone_lr_scale
        
        # Create parameter groups
        param_groups = [
            {'params': backbone_params, 'lr': cfg.learning_rate * lr_scale},
            {'params': head_params, 'lr': cfg.learning_rate}
        ]
        
        logger.info("Using differential learning rates:")
        logger.info(f"  Backbone ({len(backbone_params)} params): {cfg.learning_rate * lr_scale:.6f}")
        logger.info(f"  Head ({len(head_params)} params): {cfg.learning_rate:.6f}")
        
        return torch.optim.Adam(param_groups, weight_decay=cfg.weight_decay)
    
    else:
        # Standard optimizer
        params_to_update = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Standard optimizer: {len(params_to_update)} parameters, lr={cfg.learning_rate}")
        return torch.optim.Adam(params_to_update, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)


def add_transfer_info_to_checkpoint(checkpoint: Dict[str, Any], cfg: DictConfig) -> Dict[str, Any]:
    """
    Add transfer learning information to checkpoint if applicable.
    
    Args:
        checkpoint: Checkpoint dictionary
        cfg: Configuration
        
    Returns:
        Updated checkpoint
    """
    if hasattr(cfg, 'transfer_learning') and cfg.transfer_learning.enabled:
        checkpoint['transfer_learning_info'] = {
            'source_checkpoint': cfg.transfer_learning.source_checkpoint,
            'source_variant': cfg.transfer_learning.source_variant,
            'target_variant': cfg.model.params.variant,
            'method': cfg.transfer_learning.method,
            'frozen_layers': getattr(cfg.transfer_learning, 'freeze_layers', 0)
        }
    return checkpoint