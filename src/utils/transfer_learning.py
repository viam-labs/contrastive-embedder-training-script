import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch.nn.utils.prune as prune
import logging

# Set up a logger instance for the module
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('osnet_transfer')

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

# The helper function remains the same, as it was already well-defined.
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