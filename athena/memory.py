"""
Cross-Task Memory implementation for rehearsal-based continual learning.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class CrossTaskMemory:
    """
    Memory module for storing and retrieving examples from previous tasks.
    
    Args:
        model: The model to extract features from
        memory_size: Maximum number of examples to store
        feature_dim: Dimension of feature vectors
        num_neighbors: Number of nearest neighbors to retrieve
        temperature: Temperature for softmax in retrieval
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_size: int = 1000,
        feature_dim: int = 768,
        num_neighbors: int = 5,
        temperature: float = 0.1,
    ):
        self.model = model
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.num_neighbors = num_neighbors
        self.temperature = temperature
        
        # Initialize memory
        self.features = []
        self.examples = []
        self.task_ids = []
        
        # Initialize nearest neighbor search
        self.nn = NearestNeighbors(
            n_neighbors=min(num_neighbors, memory_size),
            metric="cosine"
        )
    
    def _extract_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract features from a batch using the model."""
        with torch.no_grad():
            outputs = self.model(**batch, output_hidden_states=True)
            # Use last hidden state as features
            features = outputs.hidden_states[-1][:, 0]  # [CLS] token
        return features
    
    def add_examples(
        self,
        batch: Dict[str, torch.Tensor],
        task_id: int,
        max_examples: Optional[int] = None
    ):
        """
        Add examples to memory.
        
        Args:
            batch: Batch of examples
            task_id: ID of the task
            max_examples: Maximum number of examples to add per task
        """
        features = self._extract_features(batch)
        
        # Convert to numpy for storage
        features_np = features.cpu().numpy()
        
        # Add to memory
        self.features.extend(features_np)
        self.examples.extend([batch] * len(features_np))
        self.task_ids.extend([task_id] * len(features_np))
        
        # Trim memory if needed
        if len(self.features) > self.memory_size:
            # Remove oldest examples
            self.features = self.features[-self.memory_size:]
            self.examples = self.examples[-self.memory_size:]
            self.task_ids = self.task_ids[-self.memory_size:]
        
        # Update nearest neighbor search
        self.nn.fit(np.array(self.features))
    
    def get_neighbors(
        self,
        batch: Dict[str, torch.Tensor],
        task_id: Optional[int] = None
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[float]]:
        """
        Retrieve nearest neighbors for a batch.
        
        Args:
            batch: Batch of examples
            task_id: Optional task ID to filter neighbors
            
        Returns:
            Tuple of (neighbor examples, similarity scores)
        """
        features = self._extract_features(batch)
        features_np = features.cpu().numpy()
        
        # Get nearest neighbors
        distances, indices = self.nn.kneighbors(features_np)
        
        # Convert distances to similarities
        similarities = np.exp(-distances / self.temperature)
        
        # Filter by task ID if specified
        if task_id is not None:
            filtered_indices = []
            filtered_similarities = []
            for i, idx in enumerate(indices):
                task_mask = np.array(self.task_ids)[idx] == task_id
                filtered_indices.append(idx[task_mask])
                filtered_similarities.append(similarities[i][task_mask])
            indices = filtered_indices
            similarities = filtered_similarities
        
        # Get examples
        neighbor_examples = []
        for idx in indices:
            neighbor_examples.append([self.examples[i] for i in idx])
        
        return neighbor_examples, similarities
    
    def get_rehearsal_batch(
        self,
        batch: Dict[str, torch.Tensor],
        task_id: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get a batch of examples for rehearsal.
        
        Args:
            batch: Current batch
            task_id: Optional task ID to filter neighbors
            
        Returns:
            Combined batch with current and rehearsal examples
        """
        neighbor_examples, similarities = self.get_neighbors(batch, task_id)
        
        # Combine current batch with neighbors
        combined_batch = {}
        for key in batch.keys():
            if isinstance(batch[key], torch.Tensor):
                # Stack current batch with neighbors
                neighbor_tensors = [
                    torch.stack([ex[key] for ex in neighbors])
                    for neighbors in neighbor_examples
                ]
                combined_batch[key] = torch.cat([
                    batch[key].unsqueeze(1),
                    *neighbor_tensors
                ], dim=1)
        
        return combined_batch
    
    def clear(self):
        """Clear memory."""
        self.features = []
        self.examples = []
        self.task_ids = []
        self.nn = NearestNeighbors(
            n_neighbors=min(self.num_neighbors, self.memory_size),
            metric="cosine"
        )
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get memory statistics."""
        return {
            "total_examples": len(self.features),
            "unique_tasks": len(set(self.task_ids)),
            "memory_size": self.memory_size,
        } 