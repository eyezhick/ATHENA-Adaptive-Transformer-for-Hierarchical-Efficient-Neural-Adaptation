"""
PolyAdapter implementation combining LoRA, IA³, and MoE components.
"""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 32.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x @ self.lora_A) @ self.lora_B * self.scaling


class IA3Layer(nn.Module):
    """IA³ scaling layer."""
    
    def __init__(self, features: int):
        super().__init__()
        self.scaling = nn.Parameter(torch.ones(features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scaling


class MoELayer(nn.Module):
    """Mixture of Experts layer with sparse routing."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_experts: int = 4,
        k: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for _ in range(num_experts)
        ])
        
        # Router
        self.router = nn.Linear(in_features, num_experts)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get routing weights
        router_logits = self.router(x)
        router_weights = F.softmax(router_logits, dim=-1)
        
        # Top-k routing
        if self.k < self.num_experts:
            top_k_weights, top_k_indices = torch.topk(router_weights, self.k, dim=-1)
            router_weights = torch.zeros_like(router_weights).scatter_(
                -1, top_k_indices, top_k_weights
            )
        
        # Expert outputs
        expert_outputs = torch.stack([
            expert(x) for expert in self.experts
        ], dim=-2)
        
        # Weighted combination
        return torch.sum(
            expert_outputs * router_weights.unsqueeze(-1),
            dim=-2
        )


class PolyAdapter(nn.Module):
    """
    PolyAdapter layer combining LoRA, IA³, and MoE components.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        rank: LoRA rank
        num_experts: Number of experts in MoE
        k: Number of experts to route to
        dropout: Dropout probability
        use_lora: Whether to use LoRA
        use_ia3: Whether to use IA³
        use_moe: Whether to use MoE
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        num_experts: int = 4,
        k: int = 1,
        dropout: float = 0.1,
        use_lora: bool = True,
        use_ia3: bool = True,
        use_moe: bool = True,
    ):
        super().__init__()
        self.use_lora = use_lora
        self.use_ia3 = use_ia3
        self.use_moe = use_moe
        
        # Base layer (frozen)
        self.base_layer = nn.Linear(in_features, out_features)
        self.base_layer.requires_grad_(False)
        
        # Adapter components
        if use_lora:
            self.lora = LoRALayer(in_features, out_features, rank, dropout=dropout)
        if use_ia3:
            self.ia3 = IA3Layer(out_features)
        if use_moe:
            self.moe = MoELayer(in_features, out_features, num_experts, k, dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base layer output
        base_output = self.base_layer(x)
        
        # Adapter components
        output = base_output
        if self.use_lora:
            output = output + self.lora(x)
        if self.use_ia3:
            output = self.ia3(output)
        if self.use_moe:
            output = output + self.moe(x)
        
        return output
    
    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        """Get dictionary of trainable parameters."""
        params = {}
        if self.use_lora:
            params.update({
                'lora_A': self.lora.lora_A,
                'lora_B': self.lora.lora_B,
            })
        if self.use_ia3:
            params['ia3_scaling'] = self.ia3.scaling
        if self.use_moe:
            params.update({
                f'moe_expert_{i}': expert.weight
                for i, expert in enumerate(self.moe.experts)
            })
            params['moe_router'] = self.moe.router.weight
        return params
    
    @property
    def num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 