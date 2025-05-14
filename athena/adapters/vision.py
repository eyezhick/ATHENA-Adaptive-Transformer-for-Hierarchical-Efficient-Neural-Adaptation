"""
Vision-Language adapter implementation for multimodal models.
"""

import math
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel


class VisionPolyAdapter(nn.Module):
    """
    Vision-Language PolyAdapter that extends the base PolyAdapter for multimodal models.
    
    Args:
        vision_features: Vision encoder feature dimension
        text_features: Text encoder feature dimension
        rank: LoRA rank
        num_experts: Number of experts in MoE
        k: Number of experts to route to
        dropout: Dropout probability
        use_lora: Whether to use LoRA
        use_ia3: Whether to use IA³
        use_moe: Whether to use MoE
        use_cross_attention: Whether to use cross-attention
    """
    
    def __init__(
        self,
        vision_features: int,
        text_features: int,
        rank: int = 8,
        num_experts: int = 4,
        k: int = 1,
        dropout: float = 0.1,
        use_lora: bool = True,
        use_ia3: bool = True,
        use_moe: bool = True,
        use_cross_attention: bool = True,
    ):
        super().__init__()
        self.vision_features = vision_features
        self.text_features = text_features
        self.use_lora = use_lora
        self.use_ia3 = use_ia3
        self.use_moe = use_moe
        self.use_cross_attention = use_cross_attention
        
        # Cross-attention layer
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=text_features,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
        
        # Vision adapter components
        if use_lora:
            self.vision_lora = nn.ModuleDict({
                'q': nn.Linear(vision_features, rank, bias=False),
                'k': nn.Linear(vision_features, rank, bias=False),
                'v': nn.Linear(vision_features, rank, bias=False),
                'o': nn.Linear(rank, vision_features, bias=False)
            })
        
        if use_ia3:
            self.vision_ia3 = nn.ParameterDict({
                'q': nn.Parameter(torch.ones(vision_features)),
                'k': nn.Parameter(torch.ones(vision_features)),
                'v': nn.Parameter(torch.ones(vision_features))
            })
        
        if use_moe:
            self.vision_moe = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(vision_features, vision_features),
                    nn.GELU(),
                    nn.Linear(vision_features, vision_features)
                ) for _ in range(num_experts)
            ])
            self.vision_router = nn.Linear(vision_features, num_experts)
        
        # Text adapter components
        if use_lora:
            self.text_lora = nn.ModuleDict({
                'q': nn.Linear(text_features, rank, bias=False),
                'k': nn.Linear(text_features, rank, bias=False),
                'v': nn.Linear(text_features, rank, bias=False),
                'o': nn.Linear(rank, text_features, bias=False)
            })
        
        if use_ia3:
            self.text_ia3 = nn.ParameterDict({
                'q': nn.Parameter(torch.ones(text_features)),
                'k': nn.Parameter(torch.ones(text_features)),
                'v': nn.Parameter(torch.ones(text_features))
            })
        
        if use_moe:
            self.text_moe = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(text_features, text_features),
                    nn.GELU(),
                    nn.Linear(text_features, text_features)
                ) for _ in range(num_experts)
            ])
            self.text_router = nn.Linear(text_features, num_experts)
        
        self.dropout = nn.Dropout(dropout)
    
    def _apply_lora(
        self,
        x: torch.Tensor,
        lora_dict: nn.ModuleDict,
        ia3_dict: Optional[nn.ParameterDict] = None
    ) -> torch.Tensor:
        """Apply LoRA and IA³ to input."""
        q = lora_dict['q'](x)
        k = lora_dict['k'](x)
        v = lora_dict['v'](x)
        
        if ia3_dict is not None:
            q = q * ia3_dict['q']
            k = k * ia3_dict['k']
            v = v * ia3_dict['v']
        
        return lora_dict['o'](q + k + v)
    
    def _apply_moe(
        self,
        x: torch.Tensor,
        moe_list: nn.ModuleList,
        router: nn.Linear,
        k: int
    ) -> torch.Tensor:
        """Apply MoE to input."""
        router_logits = router(x)
        router_weights = F.softmax(router_logits, dim=-1)
        
        if k < len(moe_list):
            top_k_weights, top_k_indices = torch.topk(router_weights, k, dim=-1)
            router_weights = torch.zeros_like(router_weights).scatter_(
                -1, top_k_indices, top_k_weights
            )
        
        expert_outputs = torch.stack([
            expert(x) for expert in moe_list
        ], dim=-2)
        
        return torch.sum(
            expert_outputs * router_weights.unsqueeze(-1),
            dim=-2
        )
    
    def forward(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            vision_features: Vision encoder features [batch_size, seq_len, vision_features]
            text_features: Text encoder features [batch_size, seq_len, text_features]
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (updated vision features, updated text features)
        """
        # Apply cross-attention if enabled
        if self.use_cross_attention:
            vision_attended, _ = self.cross_attention(
                query=text_features,
                key=vision_features,
                value=vision_features,
                key_padding_mask=attention_mask
            )
            text_attended, _ = self.cross_attention(
                query=vision_features,
                key=text_features,
                value=text_features,
                key_padding_mask=attention_mask
            )
        else:
            vision_attended = vision_features
            text_attended = text_features
        
        # Apply vision adapters
        vision_output = vision_attended
        if self.use_lora:
            vision_output = vision_output + self._apply_lora(
                vision_attended,
                self.vision_lora,
                self.vision_ia3 if self.use_ia3 else None
            )
        if self.use_moe:
            vision_output = vision_output + self._apply_moe(
                vision_attended,
                self.vision_moe,
                self.vision_router,
                self.k
            )
        
        # Apply text adapters
        text_output = text_attended
        if self.use_lora:
            text_output = text_output + self._apply_lora(
                text_attended,
                self.text_lora,
                self.text_ia3 if self.use_ia3 else None
            )
        if self.use_moe:
            text_output = text_output + self._apply_moe(
                text_attended,
                self.text_moe,
                self.text_router,
                self.k
            )
        
        return vision_output, text_output
    
    def get_trainable_params(self) -> Dict[str, torch.Tensor]:
        """Get dictionary of trainable parameters."""
        params = {}
        
        if self.use_cross_attention:
            params['cross_attention'] = self.cross_attention.state_dict()
        
        if self.use_lora:
            params.update({
                'vision_lora': self.vision_lora.state_dict(),
                'text_lora': self.text_lora.state_dict()
            })
        
        if self.use_ia3:
            params.update({
                'vision_ia3': self.vision_ia3.state_dict(),
                'text_ia3': self.text_ia3.state_dict()
            })
        
        if self.use_moe:
            params.update({
                'vision_moe': self.vision_moe.state_dict(),
                'vision_router': self.vision_router.state_dict(),
                'text_moe': self.text_moe.state_dict(),
                'text_router': self.text_router.state_dict()
            })
        
        return params
    
    @property
    def num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad) 