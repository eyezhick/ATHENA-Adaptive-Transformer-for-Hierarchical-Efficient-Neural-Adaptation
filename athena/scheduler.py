"""
Progressive Freezing Scheduler for dynamic layer freezing.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


class ProgressiveFreezingScheduler:
    """
    Scheduler that progressively freezes layers based on gradient convergence.
    
    Args:
        model: The model to schedule
        threshold: Gradient norm threshold for freezing
        window_size: Number of steps to check for convergence
        unfreeze_threshold: Loss increase threshold for unfreezing
        min_steps: Minimum number of steps before freezing starts
    """
    
    def __init__(
        self,
        model: nn.Module,
        threshold: float = 1e-5,
        window_size: int = 10,
        unfreeze_threshold: float = 0.1,
        min_steps: int = 100,
    ):
        self.model = model
        self.threshold = threshold
        self.window_size = window_size
        self.unfreeze_threshold = unfreeze_threshold
        self.min_steps = min_steps
        
        # Get trainable layers
        self.layers = self._get_trainable_layers()
        
        # Initialize state
        self.step = 0
        self.frozen_layers = set()
        self.grad_history = {i: [] for i in range(len(self.layers))}
        self.best_loss = float("inf")
    
    def _get_trainable_layers(self) -> List[nn.Module]:
        """Get list of trainable layers."""
        layers = []
        for module in self.model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                layers.append(module)
        return layers
    
    def _compute_grad_norm(self, layer: nn.Module) -> float:
        """Compute gradient norm for a layer."""
        grad_norm = 0.0
        for param in layer.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm(2).item() ** 2
        return grad_norm ** 0.5
    
    def _check_convergence(self, layer_idx: int) -> bool:
        """Check if a layer has converged based on gradient history."""
        if len(self.grad_history[layer_idx]) < self.window_size:
            return False
        
        recent_grads = self.grad_history[layer_idx][-self.window_size:]
        return np.mean(recent_grads) < self.threshold
    
    def _should_unfreeze(self, current_loss: float) -> bool:
        """Check if any frozen layers should be unfrozen."""
        return current_loss > self.best_loss * (1 + self.unfreeze_threshold)
    
    def step(self, current_loss: float) -> Dict[str, bool]:
        """
        Update layer freezing state based on current training step.
        
        Args:
            current_loss: Current training loss
            
        Returns:
            Dictionary indicating which layers were frozen/unfrozen
        """
        self.step += 1
        changes = {}
        
        # Update best loss
        if current_loss < self.best_loss:
            self.best_loss = current_loss
        
        # Check for unfreezing if loss has increased significantly
        if self._should_unfreeze(current_loss):
            for layer_idx in list(self.frozen_layers):
                self.layers[layer_idx].requires_grad_(True)
                self.frozen_layers.remove(layer_idx)
                changes[f"unfroze_layer_{layer_idx}"] = True
        
        # Skip freezing check if minimum steps not reached
        if self.step < self.min_steps:
            return changes
        
        # Update gradient history and check for freezing
        for i, layer in enumerate(self.layers):
            if i in self.frozen_layers:
                continue
            
            grad_norm = self._compute_grad_norm(layer)
            self.grad_history[i].append(grad_norm)
            
            if self._check_convergence(i):
                layer.requires_grad_(False)
                self.frozen_layers.add(i)
                changes[f"froze_layer_{i}"] = True
        
        return changes
    
    def get_frozen_layers(self) -> List[int]:
        """Get list of currently frozen layer indices."""
        return list(self.frozen_layers)
    
    def get_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(
            p.numel() for p in self.model.parameters()
            if p.requires_grad
        )
    
    def reset(self):
        """Reset scheduler state."""
        self.step = 0
        self.frozen_layers.clear()
        self.grad_history = {i: [] for i in range(len(self.layers))}
        self.best_loss = float("inf")
        
        # Unfreeze all layers
        for layer in self.layers:
            layer.requires_grad_(True) 