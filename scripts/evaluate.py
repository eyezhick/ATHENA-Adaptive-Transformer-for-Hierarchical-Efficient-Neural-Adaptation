#!/usr/bin/env python3
"""
Evaluation script for ATHENA models.
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
from athena import PolyAdapter
from athena.utils import setup_logging

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ATHENA model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="eval_outputs", help="Output directory")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def prepare_model(config, checkpoint_path):
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16 if config["training"]["mixed_precision"] == "bf16" else torch.float16,
        use_flash_attention_2=config["model"]["use_flash_attention"],
    )
    
    # Add PolyAdapter layers if not already present
    if not hasattr(model.transformer.h[0].attention, "is_polyadapter"):
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
    
    return dataset

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Setup logging
    setup_logging()
    
    # Prepare model and dataset
    model = prepare_model(config, args.checkpoint)
    dataset = prepare_dataset(config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    # Setup evaluation arguments
    eval_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_eval_batch_size=config["training"]["batch_size"],
        fp16=config["training"]["mixed_precision"] == "fp16",
        bf16=config["training"]["mixed_precision"] == "bf16",
        report_to="wandb" if config["logging"]["wandb"] else None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
    )
    
    # Run evaluation
    metrics = trainer.evaluate()
    
    # Print metrics
    print("\nEvaluation Results:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Save metrics
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

if __name__ == "__main__":
    main() 