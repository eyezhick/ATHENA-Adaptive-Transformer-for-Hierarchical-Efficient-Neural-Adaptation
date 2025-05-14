"""
Evaluation script for vision-language models using ATHENA.
"""

import os
import argparse
import logging
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForVision2Seq
from datasets import load_dataset
import evaluate
from tqdm.auto import tqdm

from athena.adapters.vision import VisionPolyAdapter
from athena.adapters.qformer import QFormerAdapter
from athena.utils import load_config, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate vision-language models with ATHENA")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="eval_outputs", help="Output directory")
    return parser.parse_args()


def prepare_dataset(config: Dict[str, Any], processor: Any):
    """Prepare dataset for evaluation."""
    dataset = load_dataset(
        config["dataset"]["name"],
        split=config["dataset"]["test_split"]
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


def evaluate_model(
    model: nn.Module,
    processor: Any,
    eval_dataloader: DataLoader,
    config: Dict[str, Any],
    output_dir: str
):
    """Evaluate the model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Initialize metrics
    metrics = {}
    for metric_name in config["evaluation"]["metrics"]:
        metrics[metric_name] = evaluate.load(metric_name)
    
    # Initialize results
    results = {
        "predictions": [],
        "references": [],
        "metrics": {}
    }
    
    # Evaluation loop
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Generate predictions
            generated_ids = model.generate(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=config["generation"]["max_length"],
                num_beams=config["generation"]["num_beams"],
                temperature=config["generation"]["temperature"],
                top_p=config["generation"]["top_p"],
                repetition_penalty=config["generation"]["repetition_penalty"],
                length_penalty=config["generation"]["length_penalty"],
                no_repeat_ngram_size=config["generation"]["no_repeat_ngram_size"],
                early_stopping=config["generation"]["early_stopping"]
            )
            
            # Decode predictions and references
            predictions = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )
            references = processor.batch_decode(
                batch["labels"],
                skip_special_tokens=True
            )
            
            results["predictions"].extend(predictions)
            results["references"].extend(references)
    
    # Compute metrics
    for metric_name, metric in metrics.items():
        results["metrics"][metric_name] = metric.compute(
            predictions=results["predictions"],
            references=results["references"]
        )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save predictions and references
    with open(os.path.join(output_dir, "predictions.txt"), "w") as f:
        for pred in results["predictions"]:
            f.write(pred + "\n")
    
    with open(os.path.join(output_dir, "references.txt"), "w") as f:
        for ref in results["references"]:
            f.write(ref + "\n")
    
    # Save metrics
    with open(os.path.join(output_dir, "metrics.txt"), "w") as f:
        for metric_name, metric_value in results["metrics"].items():
            f.write(f"{metric_name}: {metric_value}\n")
    
    return results


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
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
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Prepare dataset
    eval_dataset = prepare_dataset(config, processor)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
        num_workers=config["evaluation"]["num_workers"]
    )
    
    # Evaluate
    results = evaluate_model(
        model=model,
        processor=processor,
        eval_dataloader=eval_dataloader,
        config=config,
        output_dir=args.output_dir
    )
    
    # Log results
    logger.info("Evaluation results:")
    for metric_name, metric_value in results["metrics"].items():
        logger.info(f"{metric_name}: {metric_value}")


if __name__ == "__main__":
    main() 