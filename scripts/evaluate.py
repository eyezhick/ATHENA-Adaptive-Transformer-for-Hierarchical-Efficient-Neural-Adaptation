#!/usr/bin/env python3
"""
ATHENA evaluation script.
"""

import os
import sys
import argparse
from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from athena.utils import load_config, count_trainable_params, compute_flops


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ATHENA model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="eval_outputs",
        help="Output directory",
    )
    return parser.parse_args()


def evaluate_model(
    model: nn.Module,
    eval_dataloader: DataLoader,
    config: Dict,
    output_dir: str,
):
    """
    Evaluate model on test set.
    
    Args:
        model: HuggingFace model
        eval_dataloader: Evaluation dataloader
        config: Evaluation configuration
        output_dir: Output directory
    """
    # Initialize metrics
    metrics = {}
    if config["evaluation"]["metrics"]:
        for metric_name in config["evaluation"]["metrics"]:
            metrics[metric_name] = evaluate.load(metric_name)
    
    # Evaluation loop
    model.eval()
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Generate predictions
            outputs = model.generate(
                **batch,
                **config["generation"],
            )
            
            # Decode predictions and references
            predictions = model.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True,
            )
            references = model.tokenizer.batch_decode(
                batch["labels"],
                skip_special_tokens=True,
            )
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Compute metrics
    results = {}
    for metric_name, metric in metrics.items():
        results[metric_name] = metric.compute(
            predictions=all_predictions,
            references=all_references,
        )
    
    # Add model stats
    results["model_stats"] = {
        "trainable_params": count_trainable_params(model),
        "total_params": sum(p.numel() for p in model.parameters()),
        "flops": compute_flops(
            model=model,
            input_shape=(
                config["model"]["max_length"],
                config["model"]["hidden_size"],
            ),
            batch_size=config["evaluation"]["batch_size"],
        ),
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch.float16 if config["evaluation"]["use_fp16"] else torch.float32,
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Load dataset
    # TODO: Implement dataset loading based on config
    
    # Evaluate model
    results = evaluate_model(
        model=model,
        eval_dataloader=eval_dataloader,
        config=config,
        output_dir=args.output_dir,
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for metric_name, value in results.items():
        if isinstance(value, dict):
            print(f"\n{metric_name}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{metric_name}: {value}")


if __name__ == "__main__":
    main() 