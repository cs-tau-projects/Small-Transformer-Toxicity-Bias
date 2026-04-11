import os
import argparse
import torch
from transformers import set_seed
from src.steps.utils import setup_logging

def main():
    parser = argparse.ArgumentParser(description="Toxicity Bias Evaluation Pipeline")
    parser.add_argument("--step", type=str, default="all",
                        choices=["data", "baseline", "eval-raw", "finetune", "eval-finetuned", "eval-ood", "llama", "analysis", "report", "all"],
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

    # 1. Pipeline Logging and Reproducibility
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logging(args.output_dir)
    logger.info("Initializing Toxicity Bias Evaluation Pipeline")

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
    if args.step in ["eval-raw", "finetune", "eval-finetuned", "eval-ood", "llama", "all"]:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using device: [bold green]{device}[/bold green] (GPU Count: {torch.cuda.device_count()})", extra={"markup": True})
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info(f"Using device: [bold blue]{device}[/bold blue] (Metal)", extra={"markup": True})
        else:
            device = torch.device("cpu")
            logger.warning(" [bold red]CUDA is NOT available.[/bold red] PyTorch cannot see your GPU.", extra={"markup": True})
            logger.info(f"Using device: [bold yellow]{device}[/bold yellow]", extra={"markup": True})
    
    if args.step in ["data", "all"]:
        from src.steps.data_step import run_data_step
        logger.info("Running [bold cyan]Data Strategy[/bold cyan] step", extra={"markup": True})
        run_data_step(cache_dir, data_dir, args.train_samples, args.eval_samples, seed=args.seed)

    if args.step in ["baseline", "all"]:
        from src.steps.baseline_step import run_baseline_step
        logger.info("Running [bold cyan]Baseline[/bold cyan] step", extra={"markup": True})
        run_baseline_step(data_dir, results_dir)

    if args.step == "eval-raw":
        from src.steps.eval_raw_step import run_eval_raw_step
        logger.info("Running [bold cyan]Raw Evaluation[/bold cyan] step", extra={"markup": True})
        run_eval_raw_step(data_dir, results_dir, cache_dir, args.models, device)

    if args.step in ["finetune", "all"]:
        from src.steps.finetune_step import run_finetune_step
        logger.info("Running [bold cyan]Fine-tuning[/bold cyan] step", extra={"markup": True})
        run_finetune_step(args.models, args.output_dir, cache_dir=cache_dir, seed=args.seed, train_samples=args.train_samples, data_dir=data_dir)

    if args.step in ["eval-finetuned", "all"]:
        from src.steps.eval_ft_step import run_eval_ft_step
        logger.info("Running [bold cyan]Fine-tuned Evaluation[/bold cyan] step", extra={"markup": True})
        run_eval_ft_step(data_dir, results_dir, cache_dir, args.output_dir, args.models, device)

    if args.step in ["eval-ood", "all"]:
        from src.steps.eval_ood_step import run_eval_ood_step
        logger.info("Running [bold cyan]OOD Evaluation[/bold cyan] step", extra={"markup": True})
        run_eval_ood_step(results_dir, cache_dir, args.output_dir, args.models, device, args.eval_samples, seed=args.seed)

    if args.step in ["llama", "all"]:
        from src.steps.llama_step import run_llama_step
        logger.info("Running [bold cyan]LLaMA Evaluation[/bold cyan] step", extra={"markup": True})
        run_llama_step(data_dir, results_dir, cache_dir, args.llama_model, device)

    if args.step in ["analysis", "all"]:
        from src.analysis import run_analysis_step
        logger.info("Running [bold cyan]Dataset & Error Analysis[/bold cyan] step", extra={"markup": True})
        # Default to the first model in the list for error analysis if none specific is provided
        primary_model = args.models[0] if args.models else None
        run_analysis_step(data_dir, results_dir, model_name_for_errors=primary_model)

    if args.step in ["report", "all"]:
        from src.steps.report_step import run_report_step
        logger.info("Generating [bold cyan]Final Report[/bold cyan]", extra={"markup": True})
        run_report_step(data_dir, results_dir, cache_dir, args.llama_model, args.models, args.eval_samples, seed=args.seed)

if __name__ == "__main__":
    main()
