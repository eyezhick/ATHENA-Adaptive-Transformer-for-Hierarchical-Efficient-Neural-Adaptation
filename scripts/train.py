#!/usr/bin/env python3
"""
Training script for ATHENA models.
"""

import argparse
import os
import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from athena import PolyAdapter, AutoRank, ProgressiveFreezingScheduler
from athena.utils import setup_logging, setup_distributed

def parse_args():
    parser = argparse.ArgumentParser(description="Train ATHENA model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--distributed", action="store_true", help="Use distributed training")
    parser.add_argument("--deepspeed", type=str, help="Path to DeepSpeed config")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def prepare_model(config):
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.bfloat16 if config["training"]["mixed_precision"] == "bf16" else torch.float16,
        use_flash_attention_2=config["model"]["use_flash_attention"],
    )
    
    # Add PolyAdapter layers
    for layer in model.transformer.h:
        layer.attention = PolyAdapter(
            in_features=model.config.hidden_size,
            out_features=model.config.hidden_size,
            rank=config["adapter"]["rank"],
            num_experts=config["adapter"]["num_experts"],
            use_lora=config["adapter"]["use_lora"],
            use_ia3=config["adapter"]["use_ia3"],
            use_moe=config["adapter"]["use_moe"],
        )
    
    return model

def prepare_dataset(config):
    dataset = load_dataset(config["dataset"]["name"])
    
    if config["dataset"]["max_samples"]:
        dataset = dataset.select(range(min(config["dataset"]["max_samples"], len(dataset))))
    
    if config["dataset"]["validation_split"]:
        dataset = dataset.train_test_split(test_size=config["dataset"]["validation_split"])
    
    return dataset

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Setup logging and distributed training
    setup_logging()
    if args.distributed:
        setup_distributed()
    
    # Prepare model and dataset
    model = prepare_model(config)
    dataset = prepare_dataset(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    # Initialize AutoRank optimizer
    autorank = AutoRank(
        model=model,
        rank_budget=config["autorank"]["rank_budget"],
        num_trials=config["autorank"]["num_trials"],
    )
    
    # Initialize progressive freezing scheduler
    freezing_scheduler = ProgressiveFreezingScheduler(
        model=model,
        threshold=config["freezing"]["threshold"],
        window_size=config["freezing"]["window_size"],
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config["training"]["batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        learning_rate=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        warmup_steps=config["training"]["warmup_steps"],
        max_steps=config["training"]["max_steps"],
        fp16=config["training"]["mixed_precision"] == "fp16",
        bf16=config["training"]["mixed_precision"] == "bf16",
        gradient_checkpointing=config["training"]["gradient_checkpointing"],
        logging_steps=config["logging"]["log_steps"],
        eval_steps=config["logging"]["eval_steps"],
        save_steps=config["logging"]["save_steps"],
        report_to="wandb" if config["logging"]["wandb"] else None,
        deepspeed=args.deepspeed,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
    )
    
    # Train model
    trainer.train()
    
    # Save final model
    trainer.save_model(os.path.join(args.output_dir, "final"))

if __name__ == "__main__":
    main() 