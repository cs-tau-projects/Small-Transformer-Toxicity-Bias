import os
import argparse
import random
import numpy as np
import torch
from transformers import set_seed

def main():
    parser = argparse.ArgumentParser(description="Toxicity Bias Evaluation Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["data", "baseline", "eval-raw", "finetune", "eval-finetuned", "llama", "report", "all"],
                        help="Which step of the pipeline to run.")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Base directory for caches, models, and outputs.")
    parser.add_argument("--models", type=str, nargs="+", 
                        default=["distilbert-base-uncased", "distilroberta-base", "google/bert_uncased_L-4_H-512_A-8"],
                        help="List of transformer models to evaluate.")
    parser.add_argument("--llama_model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="LLaMA model identifier for inference step.")
    parser.add_argument("--train_samples", type=int, default=20000,
                        help="Number of training samples to use for baseline (-1 for all).")
    parser.add_argument("--eval_samples", type=int, default=5000,
                        help="Number of evaluation samples to use (-1 for all).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed.")
    args = parser.parse_args()

    # 1. Global Reproducibility
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Shared directories
    cache_dir = os.path.join(args.output_dir, ".cache")
    data_dir = os.path.join(args.output_dir, "data")
    results_dir = os.path.join(args.output_dir, "results")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # We only determine the device if we are running a model-related step
    device = None
    if args.step in ["eval-raw", "finetune", "eval-finetuned", "llama", "all"]:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        print(f"Using device: {device}")
    
    if args.step in ["data", "all"]:
        from src.steps.data_step import run_data_step
        run_data_step(cache_dir, data_dir, args.train_samples, args.eval_samples)

    if args.step in ["baseline", "all"]:
        from src.steps.baseline_step import run_baseline_step
        run_baseline_step(data_dir, results_dir)

    if args.step in ["eval-raw", "all"]:
        from src.steps.eval_raw_step import run_eval_raw_step
        run_eval_raw_step(data_dir, results_dir, cache_dir, args.models, device)

    if args.step in ["finetune", "all"]:
        from src.steps.finetune_step import run_finetune_step
        # Note: finetune_step delegates to src.train, so it does not need the 'device' variable.
        # It also does not use args.train_samples since src.train manages its own dataset splits.
        run_finetune_step(args.models, args.output_dir, seed=args.seed)

    if args.step in ["eval-finetuned", "all"]:
        from src.steps.eval_ft_step import run_eval_ft_step
        run_eval_ft_step(data_dir, results_dir, cache_dir, args.output_dir, args.models, device)

    if args.step in ["llama", "all"]:
        from src.steps.llama_step import run_llama_step
        run_llama_step(data_dir, results_dir, cache_dir, args.llama_model, device)

    if args.step in ["report", "all"]:
        from src.steps.report_step import run_report_step
        run_report_step(results_dir, args.llama_model, args.models)

if __name__ == "__main__":
    main()
