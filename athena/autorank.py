"""
AutoRank implementation for Bayesian optimization of layer ranks.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.optuna import OptunaSearch


class AutoRank:
    """
    Bayesian optimization for finding optimal layer ranks.
    
    Args:
        model: The model to optimize
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        metric: Metric to optimize (e.g., 'loss', 'accuracy')
        mode: Optimization mode ('min' or 'max')
        rank_budget: Total rank budget across all layers
        rank_candidates: List of possible rank values
        num_trials: Number of optimization trials
        patience: Early stopping patience
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        metric: str = "loss",
        mode: str = "min",
        rank_budget: int = 1024,
        rank_candidates: List[int] = [0, 2, 4, 8, 16],
        num_trials: int = 30,
        patience: int = 5,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.metric = metric
        self.mode = mode
        self.rank_budget = rank_budget
        self.rank_candidates = rank_candidates
        self.num_trials = num_trials
        self.patience = patience
        
        # Get trainable layers
        self.layers = self._get_trainable_layers()
        
        # Initialize search space
        self.search_space = self._create_search_space()
    
    def _get_trainable_layers(self) -> List[torch.nn.Module]:
        """Get list of trainable layers that can be adapted."""
        layers = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d)):
                layers.append(module)
        return layers
    
    def _create_search_space(self) -> Dict:
        """Create search space for Ray Tune."""
        return {
            f"layer_{i}_rank": tune.choice(self.rank_candidates)
            for i in range(len(self.layers))
        }
    
    def _validate_rank_config(self, config: Dict) -> bool:
        """Validate if rank configuration meets budget constraint."""
        total_rank = sum(config.values())
        return total_rank <= self.rank_budget
    
    def _train_and_evaluate(self, config: Dict) -> float:
        """Train and evaluate model with given rank configuration."""
        # Apply rank configuration
        for i, layer in enumerate(self.layers):
            rank = config[f"layer_{i}_rank"]
            # Update layer's LoRA rank
            if hasattr(layer, "lora"):
                layer.lora.rank = rank
        
        # Train model
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters())
        
        best_val_metric = float("inf") if self.mode == "min" else float("-inf")
        patience_counter = 0
        
        for epoch in range(10):  # Max 10 epochs
            # Training
            for batch in self.train_dataloader:
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
            
            # Validation
            val_metrics = self._evaluate()
            val_metric = val_metrics[self.metric]
            
            # Early stopping
            if self.mode == "min":
                if val_metric < best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
            else:
                if val_metric > best_val_metric:
                    best_val_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
            
            if patience_counter >= self.patience:
                break
        
        return best_val_metric
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()
        metrics = {self.metric: 0.0}
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                outputs = self.model(**batch)
                metrics[self.metric] += outputs.loss.item()
        
        metrics[self.metric] /= len(self.val_dataloader)
        return metrics
    
    def optimize(self) -> Dict[str, int]:
        """
        Run Bayesian optimization to find optimal layer ranks.
        
        Returns:
            Dictionary mapping layer indices to optimal ranks
        """
        # Initialize search algorithm
        search_alg = BayesOptSearch(
            metric=self.metric,
            mode=self.mode,
            utility_kwargs={
                "kind": "ucb",
                "kappa": 2.5,
                "xi": 0.0
            }
        )
        
        # Initialize scheduler
        scheduler = ASHAScheduler(
            metric=self.metric,
            mode=self.mode,
            max_t=10,  # Max epochs
            grace_period=1,
            reduction_factor=2
        )
        
        # Run optimization
        analysis = tune.run(
            self._train_and_evaluate,
            config=self.search_space,
            num_samples=self.num_trials,
            scheduler=scheduler,
            search_alg=search_alg,
            resources_per_trial={"cpu": 1, "gpu": 1},
            verbose=1
        )
        
        # Get best configuration
        best_config = analysis.get_best_config(
            metric=self.metric,
            mode=self.mode
        )
        
        # Convert to layer ranks
        layer_ranks = {
            i: best_config[f"layer_{i}_rank"]
            for i in range(len(self.layers))
        }
        
        return layer_ranks 