"""
Training script for vision-language models using ATHENA.
"""

import os
import argparse
import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    get_scheduler,
    set_seed
)
from datasets import load_dataset
import wandb
from tqdm.auto import tqdm

from athena.adapters.vision import VisionPolyAdapter
from athena.adapters.qformer import QFormerAdapter
from athena.autorank import AutoRank
from athena.scheduler import ProgressiveFreezingScheduler
from athena.memory import CrossTaskMemory
from athena.utils import load_config, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Train vision-language models with ATHENA")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def prepare_dataset(config: Dict[str, Any], processor: Any):
    """Prepare dataset for training."""
    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["train_split"]
    )
    
    def preprocess_function(examples):
        images = examples["image"]
        texts = examples["text"]
        
        # Process images
        pixel_values = processor(
            images=images,
            return_tensors="pt",
            padding="max_length",
            max_length=config["model"]["max_length"],
            truncation=True
        ).pixel_values
        
        # Process texts
        text_inputs = processor(
            text=texts,
            padding="max_length",
            max_length=config["model"]["max_length"],
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_inputs.input_ids,
            "attention_mask": text_inputs.attention_mask,
            "labels": text_inputs.input_ids.clone()
        }
    
    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return processed_dataset


def train(
    model: nn.Module,
    processor: Any,
    train_dataloader: DataLoader,
    eval_dataloader: Optional[DataLoader],
    config: Dict[str, Any],
    output_dir: str,
    resume: Optional[str] = None
):
    """Train the model."""
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"]
    )
    
    # Initialize scheduler
    num_training_steps = len(train_dataloader) * config["training"]["num_epochs"]
    lr_scheduler = get_scheduler(
        name=config["training"]["scheduler"],
        optimizer=optimizer,
        num_warmup_steps=config["training"]["warmup_steps"],
        num_training_steps=num_training_steps
    )
    
    # Initialize AutoRank
    autorank = AutoRank(
        model=model,
        metric=config["autorank"]["metric"],
        mode=config["autorank"]["mode"],
        rank_budget=config["autorank"]["rank_budget"],
        candidates=config["autorank"]["candidates"],
        trials=config["autorank"]["trials"],
        patience=config["autorank"]["patience"],
        optimization_steps=config["autorank"]["optimization_steps"]
    )
    
    # Initialize ProgressiveFreezingScheduler
    freezing_scheduler = ProgressiveFreezingScheduler(
        model=model,
        threshold=config["progressive_freezing"]["threshold"],
        window_size=config["progressive_freezing"]["window_size"]
    )
    
    # Initialize CrossTaskMemory if enabled
    memory = None
    if config.get("memory", {}).get("enabled", False):
        memory = CrossTaskMemory(
            size=config["memory"]["size"],
            feature_dim=config["memory"]["feature_dim"],
            num_neighbors=config["memory"]["num_neighbors"],
            temperature=config["memory"]["temperature"]
        )
    
    # Initialize wandb
    if config["logging"]["wandb"]:
        wandb.init(
            project=config["logging"]["project"],
            config=config,
            tags=config["logging"]["tags"]
        )
    
    # Load checkpoint if resuming
    start_epoch = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        if memory and "memory_state_dict" in checkpoint:
            memory.load_state_dict(checkpoint["memory_state_dict"])
    
    # Training loop
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(start_epoch, config["training"]["num_epochs"]):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            if (progress_bar.n + 1) % config["training"]["gradient_accumulation_steps"] == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Update AutoRank
            autorank.step(loss.item())
            
            # Update ProgressiveFreezingScheduler
            freezing_scheduler.step(loss.item())
            
            # Update CrossTaskMemory
            if memory:
                memory.update(
                    features=outputs.hidden_states[-1].detach(),
                    labels=batch["labels"]
                )
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Log to wandb
            if config["logging"]["wandb"]:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/epoch": epoch + 1
                })
        
        # Evaluate
        if eval_dataloader:
            eval_loss = evaluate(model, eval_dataloader, device)
            if config["logging"]["wandb"]:
                wandb.log({
                    "eval/loss": eval_loss,
                    "eval/epoch": epoch + 1
                })
        
        # Save checkpoint
        if (epoch + 1) % config["training"]["save_steps"] == 0:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": lr_scheduler.state_dict(),
            }
            if memory:
                checkpoint["memory_state_dict"] = memory.state_dict()
            
            torch.save(
                checkpoint,
                os.path.join(output_dir, f"checkpoint-{epoch + 1}.pt")
            )


def evaluate(model: nn.Module, eval_dataloader: DataLoader, device: torch.device) -> float:
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            total_loss += outputs.loss.item()
    
    return total_loss / len(eval_dataloader)


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(config["model"]["name"])
    model = AutoModelForVision2Seq.from_pretrained(config["model"]["name"])
    
    # Add adapters
    if config["model"]["name"].startswith("Salesforce/blip2"):
        adapter = QFormerAdapter(
            query_features=config["model"]["qformer_hidden_size"],
            vision_features=config["model"]["vision_features"],
            text_features=config["model"]["hidden_size"],
            rank=config["adapter"]["rank"],
            num_experts=config["adapter"]["num_experts"],
            k=config["adapter"]["k"],
            dropout=config["adapter"]["dropout"],
            use_lora=config["adapter"]["use_lora"],
            use_ia3=config["adapter"]["use_ia3"],
            use_moe=config["adapter"]["use_moe"],
            use_cross_attention=config["adapter"]["use_cross_attention"],
            num_query_tokens=config["model"]["num_query_tokens"]
        )
    else:
        adapter = VisionPolyAdapter(
            vision_features=config["model"]["vision_features"],
            text_features=config["model"]["hidden_size"],
            rank=config["adapter"]["rank"],
            num_experts=config["adapter"]["num_experts"],
            k=config["adapter"]["k"],
            dropout=config["adapter"]["dropout"],
            use_lora=config["adapter"]["use_lora"],
            use_ia3=config["adapter"]["use_ia3"],
            use_moe=config["adapter"]["use_moe"],
            use_cross_attention=config["adapter"]["use_cross_attention"]
        )
    
    model.add_adapter(adapter)
    
    # Prepare dataset
    train_dataset = prepare_dataset(config, processor)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"]
    )
    
    # Prepare evaluation dataset if specified
    eval_dataloader = None
    if config["dataset"].get("eval_split"):
        eval_dataset = load_dataset(
            config["dataset"]["name"],
            split=config["dataset"]["eval_split"]
        )
        eval_dataset = eval_dataset.map(
            lambda x: prepare_dataset(config, processor)(x),
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=config["training"]["eval_batch_size"],
            shuffle=False,
            num_workers=config["training"]["num_workers"]
        )
    
    # Train
    train(
        model=model,
        processor=processor,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        config=config,
        output_dir=args.output_dir,
        resume=args.resume
    )


if __name__ == "__main__":
    main() 