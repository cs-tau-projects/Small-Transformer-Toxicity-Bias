import os
import argparse
import yaml
import torch
from transformers import set_seed

def main():
    parser = argparse.ArgumentParser(description="Toxicity Bias Evaluation Pipeline")
    
    # Add a single argument for the config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML configuration file.")

    # Parse only the config argument first
    args, remaining_argv = parser.parse_known_args()

    # Default parameters (can be overwritten by YAML)
    defaults = {
        "step": "all",
        "output_dir": "./outputs",
        "models": ["distilbert-base-uncased", "distilroberta-base", "google/bert_uncased_L-4_H-512_A-8"],
        "llama_model": "meta-llama/Llama-3.2-1B",
        "train_samples": 20000,
        "eval_samples": 5000,
        "seed": 42
    }

    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Merge YAML config with defaults. YAML values take precedence.
        # This is a simple merge; for nested dicts, a recursive merge would be better.
        if config.get('model'):
            defaults['models'] = [config['model']['name']]
        if config.get('data'):
            defaults['train_samples'] = config['data'].get('train_batch_size', defaults['train_samples'])
        if config.get('training'):
            defaults['seed'] = config.get('seed', defaults['seed'])
        if config.get('output_dir'):
            defaults['output_dir'] = config.get('output_dir', defaults['output_dir'])

    # Re-build the parser with new defaults for the remaining arguments
    parser = argparse.ArgumentParser(description="Toxicity Bias Evaluation Pipeline")
    parser.add_argument("--step", type=str, default=defaults["step"],
                        choices=["data", "baseline", "eval-raw", "finetune", "eval-finetuned", "eval-ood", "llama", "report", "all"],
                        help="Which step of the pipeline to run.")
    parser.add_argument("--output_dir", type=str, default=defaults["output_dir"],
                        help="Base directory for caches, models, and outputs.")
    parser.add_argument("--models", type=str, nargs="+", default=defaults["models"],
                        help="List of transformer models to evaluate.")
    parser.add_argument("--llama_model", type=str, default=defaults["llama_model"],
                        help="LLaMA model identifier for inference step.")
    parser.add_argument("--train_samples", type=int, default=defaults["train_samples"],
                        help="Number of training samples to use for baseline (-1 for all).")
    parser.add_argument("--eval_samples", type=int, default=defaults["eval_samples"],
                        help="Number of evaluation samples to use (-1 for all).")
    parser.add_argument("--seed", type=int, default=defaults["seed"],
                        help="Global random seed.")

    # Parse the remaining arguments
    args = parser.parse_args(remaining_argv)
    
    # Store the original config path if it exists
    if args.config:
        args.config_path = args.config
        delattr(args, 'config')

    # 1. Global Reproducibility
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Shared directories
    cache_dir = os.path.join(args.output_dir, ".cache")
    data_dir = os.path.join(args.output_dir, "data")
    results_dir = os.path.join(.args.output_dir, "results")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Pass the full config to the steps that need it
    full_config = config if 'config' in locals() else {}

    # We only determine the device if we are running a model-related step
    device = None
    if args.step in ["eval-raw", "finetune", "eval-finetuned", "eval-ood", "llama", "all"]:
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
        run_finetune_step(args.models, args.output_dir, full_config, seed=args.seed, train_samples=args.train_samples)

    if args.step in ["eval-finetuned", "all"]:
        from src.steps.eval_ft_step import run_eval_ft_step
        run_eval_ft_step(data_dir, results_dir, cache_dir, args.output_dir, args.models, device)

    if args.step == "eval-ood":
        from src.steps.eval_ood_step import run_eval_ood_step
        run_eval_ood_step(results_dir, cache_dir, args.output_dir, args.models, device, args.eval_samples)

    if args.step in ["llama", "all"]:
        from src.steps.llama_step import run_llama_step
        run_llama_step(data_dir, results_dir, cache_dir, args.llama_model, device)

    if args.step in ["report", "all"]:
        from src.steps.report_step import run_report_step
        run_report_step(results_dir, args.llama_model, args.models, full_config)

if __name__ == "__main__":
    main()
