#!/usr/bin/env python3
"""
ATHENA training script.
"""

import os
import sys
import argparse
from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import wandb
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.adapters import PolyAdapter
from athena.autorank import AutoRank
from athena.scheduler import ProgressiveFreezingScheduler
from athena.memory import CrossTaskMemory
from athena.utils import (
    load_config,
    count_trainable_params,
    compute_flops,
    save_checkpoint,
    compute_gradient_norm,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train ATHENA model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )
    return parser.parse_args()


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    config: Dict,
    output_dir: str,
    resume_path: Optional[str] = None,
):
    """
    Train model with ATHENA components.
    
    Args:
        model: HuggingFace model
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        config: Training configuration
        output_dir: Output directory
        resume_path: Path to checkpoint to resume from
    """
    # Initialize components
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    num_training_steps = len(train_dataloader) * config["training"]["num_epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=num_training_steps,
    )
    
    # Initialize ATHENA components
    autorank = AutoRank(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **config["autorank"],
    )
    
    freezing_scheduler = ProgressiveFreezingScheduler(
        model=model,
        **config["freezing_scheduler"],
    )
    
    memory = CrossTaskMemory(
        model=model,
        **config["memory"],
    ) if config.get("use_memory", False) else None
    
    # Initialize wandb
    wandb.init(
        project="athena",
        config=config,
        name=config.get("run_name", "athena-run"),
    )
    
    # Resume from checkpoint if specified
    start_step = 0
    if resume_path:
        start_step, metrics = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            path=resume_path,
        )
        wandb.log(metrics, step=start_step)
    
    # Training loop
    model.train()
    for epoch in range(config["training"]["num_epochs"]):
        for step, batch in enumerate(tqdm(train_dataloader)):
            global_step = start_step + step
            
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Get rehearsal batch if using memory
            if memory is not None:
                batch = memory.get_rehearsal_batch(batch)
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update freezing scheduler
            freezing_changes = freezing_scheduler.step(loss.item())
            
            # Log metrics
            metrics = {
                "train/loss": loss.item(),
                "train/learning_rate": scheduler.get_last_lr()[0],
                "train/gradient_norm": compute_gradient_norm(model),
                "train/trainable_params": count_trainable_params(model),
            }
            
            if freezing_changes:
                metrics.update({
                    f"train/{k}": v
                    for k, v in freezing_changes.items()
                })
            
            wandb.log(metrics, step=global_step)
            
            # Validation
            if global_step % config["training"]["eval_steps"] == 0:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for val_batch in val_dataloader:
                        val_batch = {k: v.to(model.device) for k, v in val_batch.items()}
                        outputs = model(**val_batch)
                        val_loss += outputs.loss.item()
                val_loss /= len(val_dataloader)
                
                wandb.log({
                    "val/loss": val_loss,
                }, step=global_step)
                
                model.train()
            
            # Save checkpoint
            if global_step % config["training"]["save_steps"] == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    metrics=metrics,
                    path=os.path.join(output_dir, f"checkpoint-{global_step}.pt"),
                )
            
            # Run AutoRank optimization
            if global_step % config["autorank"]["optimize_steps"] == 0:
                layer_ranks = autorank.optimize()
                wandb.log({
                    f"autorank/layer_{i}_rank": rank
                    for i, rank in layer_ranks.items()
                }, step=global_step)


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.float16 if config["training"]["use_fp16"] else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    # Load dataset
    # TODO: Implement dataset loading based on config
    
    # Train model
    train(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        output_dir=args.output_dir,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main() 