"""
Utility functions for ATHENA.
"""

import os
import logging
import torch
import torch.distributed as dist
from typing import Dict, Optional

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

def setup_distributed():
    """Setup distributed training environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(rank)

def count_trainable_params(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_gradient_norm(model: torch.nn.Module) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    step: int,
    metrics: Dict[str, float],
    path: str,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "step": step,
            "metrics": metrics,
        },
        path,
    )

def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    path: str,
) -> tuple[int, Dict[str, float]]:
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint["step"], checkpoint["metrics"]

def compute_flops(
    model: torch.nn.Module,
    input_shape: tuple[int, int],
    batch_size: int,
) -> int:
    """Compute FLOPs for a forward pass."""
    from thop import profile
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape)
    
    # Compute FLOPs
    flops, _ = profile(model, inputs=(dummy_input,))
    return flops 