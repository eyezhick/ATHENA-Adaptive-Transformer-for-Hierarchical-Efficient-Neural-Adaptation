"""
Utility functions for ATHENA.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import json
import yaml
import torch
import torch.nn as nn
from transformers import PreTrainedModel


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r") as f:
        if ext == ".yaml" or ext == ".yml":
            config = yaml.safe_load(f)
        elif ext == ".json":
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {ext}")
    return config


def count_trainable_params(model: nn.Module) -> int:
    """
    Count number of trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size(model: nn.Module) -> int:
    """
    Get total number of parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def compute_flops(
    model: PreTrainedModel,
    input_shape: Tuple[int, int],
    batch_size: int = 1
) -> int:
    """
    Compute FLOPs for a forward pass.
    
    Args:
        model: HuggingFace model
        input_shape: (sequence_length, hidden_size)
        batch_size: Batch size
        
    Returns:
        Number of FLOPs
    """
    seq_len, hidden_size = input_shape
    
    # Count attention FLOPs
    attention_flops = (
        batch_size * seq_len * hidden_size *  # Q, K, V projections
        3 +  # 3 matrices
        batch_size * seq_len * seq_len * hidden_size *  # Attention scores
        2 +  # Multiply and softmax
        batch_size * seq_len * seq_len * hidden_size  # Output projection
    )
    
    # Count feed-forward FLOPs
    ffn_flops = (
        batch_size * seq_len * hidden_size * hidden_size * 4 *  # First layer
        2 +  # Multiply and activation
        batch_size * seq_len * hidden_size * hidden_size  # Second layer
    )
    
    # Count layer norm FLOPs
    norm_flops = batch_size * seq_len * hidden_size * 4  # Mean, var, normalize
    
    # Total FLOPs per layer
    layer_flops = attention_flops + ffn_flops + norm_flops * 2
    
    # Multiply by number of layers
    num_layers = model.config.num_hidden_layers
    total_flops = layer_flops * num_layers
    
    return total_flops


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    step: int,
    metrics: Dict[str, float],
    path: str
):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        step: Current training step
        metrics: Dictionary of metrics
        path: Path to save checkpoint
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
    }
    
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    path: str
) -> Tuple[int, Dict[str, float]]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        path: Path to checkpoint
        
    Returns:
        Tuple of (step, metrics)
    """
    checkpoint = torch.load(path)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    return checkpoint["step"], checkpoint["metrics"]


def compute_gradient_norm(model: nn.Module) -> float:
    """
    Compute total gradient norm.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total gradient norm
    """
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm(2).item() ** 2
    return total_norm ** 0.5


def compute_parameter_norm(model: nn.Module) -> float:
    """
    Compute total parameter norm.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total parameter norm
    """
    total_norm = 0.0
    for p in model.parameters():
        total_norm += p.norm(2).item() ** 2
    return total_norm ** 0.5 