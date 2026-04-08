import os
import argparse
import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed
)
from src.data.dataset import download_and_prep_jigsaw, tokenize_jigsaw_dataset, JigsawDataset
from src.data.data_utils import get_hf_token
from src.evaluator import evaluate_models_metrics

def parse_args():
    parser = argparse.ArgumentParser(description="Train DistilBERT on Jigsaw dataset.")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Model name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_base_dir", type=str, required=True, help="Base directory for output and cache.")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to pre-saved data splits (outputs/data/). When set, skips re-downloading "
                             "and uses the same splits as all other pipeline steps.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and eval.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--train_samples", type=int, default=-1,
                        help="Max training samples to use (-1 for all).")
    
    # We will compute metrics every epoch
    return parser.parse_args()

def compute_metrics(eval_pred, identity_columns, eval_dataset):
    """
    Custom compute metrics function for Hugging Face Trainer.
    Needs to tie back to the identity matrix from the dataset.
    """
    logits, labels = eval_pred
    
    # Calculate probabilities based on the number of logits
    if logits.shape[1] == 2:
        # Softmax for 2-class classification
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True)) # for numerical stability
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        probs = probs[:, 1]
    else:
        # Sigmoid for binary classification with a single output logit
        probs = 1 / (1 + np.exp(-logits[:, 0]))
    
    # To compute subgroup logic we need the identity matrix.
    # The Trainer's compute_metrics doesn't easily pass inputs, so we capture them via a closure
    # from the dataset. We pre-calculate it in the JigsawDataset object for performance.
    identity_matrix = eval_dataset.identity_matrix
    
    res_df = evaluate_models_metrics(
        y_true=labels,
        y_pred_probs=probs,
        identity_matrix=identity_matrix,
        identity_columns=identity_columns,
        threshold=0.5
    )
    
    print("\n" + "="*50)
    print("Evaluation Metrics:")
    print(res_df.to_string(index=False))
    print("="*50 + "\n")
    
    # We must return a dictionary of metrics for the Trainer
    # We use Overall AUC as the primary metric for saving the best model
    overall_auc = res_df["1. Overall AUC"].iloc[0] # taking first since overall is the same
    
    metrics_dict = {"roc_auc": overall_auc}
    for _, row in res_df.iterrows():
        ident = row["Identity"]
        if not np.isnan(row["4. Subgroup AUC"]):
            metrics_dict[f"{ident}_subgroup_auc"] = row["4. Subgroup AUC"]
        if not np.isnan(row["5. Subgroup FNR"]):
            metrics_dict[f"{ident}_subgroup_fnr"] = row["5. Subgroup FNR"]
        if not np.isnan(row["6. Subgroup FPR"]):
            metrics_dict[f"{ident}_subgroup_fpr"] = row["6. Subgroup FPR"]
            
    return metrics_dict

def main():
    args = parse_args()
    
    # 1. Reproducibility
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Setup directories
    cache_dir = os.path.join(args.output_base_dir, ".cache")
    hf_token = get_hf_token()
    output_dir = os.path.join(args.output_base_dir, "small-transformer-toxicity")
    
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using cache_dir: {cache_dir}")
    print(f"Using output_dir: {output_dir}")
    
    # 2. Load and Prepare Data
    if args.data_dir and os.path.isdir(args.data_dir):
        # --- Preferred path: reuse the splits already created by data_step ---
        # This guarantees training and evaluation see the exact same val split.
        from datasets import load_from_disk
        import json
        print(f"Loading pre-saved data splits from {args.data_dir}...")
        train_hf = load_from_disk(os.path.join(args.data_dir, "baseline_train"))
        val_hf   = load_from_disk(os.path.join(args.data_dir, "eval"))
        with open(os.path.join(args.data_dir, "identity_columns.json")) as f:
            identity_columns = json.load(f)

        # Optionally further limit training samples
        if args.train_samples > 0 and len(train_hf) > args.train_samples:
            train_hf = train_hf.select(range(args.train_samples))

        train_split = train_hf
        val_split   = val_hf
    else:
        # --- Fallback path (standalone usage without data_step) ---
        train_hf, identity_columns = download_and_prep_jigsaw("train", cache_dir=cache_dir)
        train_hf = train_hf.shuffle(seed=args.seed)

        # Apply train_samples limit before splitting
        if args.train_samples > 0 and len(train_hf) > args.train_samples:
            train_hf = train_hf.select(range(args.train_samples))

        # Let's use 10% for validation
        split_idx = int(0.9 * len(train_hf))
        train_split = train_hf.select(range(split_idx))
        val_split   = train_hf.select(range(split_idx, len(train_hf)))

    # Tokenize
    train_tokenized = tokenize_jigsaw_dataset(train_split, args.model_name, cache_dir=cache_dir)
    val_tokenized = tokenize_jigsaw_dataset(val_split, args.model_name, cache_dir=cache_dir)
    
    # Create PyTorch datasets
    train_dataset = JigsawDataset(train_tokenized, identity_columns)
    val_dataset = JigsawDataset(val_tokenized, identity_columns)
    
    # 3. Load Model
    print(f"Loading Model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        cache_dir=cache_dir,
        token=hf_token
    )
    
    # 4. Define Training Arguments and Strategy
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",  # Evaluate linearly over epochs
        save_strategy="epoch",        # Save linearly over epochs
        save_total_limit=2,           # Per instructions, do not blow up storage
        load_best_model_at_end=True,  # Per instructions
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        seed=args.seed,
        fp16=torch.cuda.is_available(), # use mixed precision if GPU available
        logging_steps=10,               # Log more frequently
        log_level="error",              # Suppress per-step loss/grad_norm prints
        disable_tqdm=False              # Explicitly ensure Trainer progress bar is enabled
    )
    
    # Create closure for compute_metrics to pass val_dataset
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, identity_columns, val_dataset)
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics_wrapper
    )
    
    # 6. Train!
    print("Starting training...")
    trainer.train()
    
    print(f"Training complete. Best model loaded from checkpont.")
    
    # 7. Final Evaluation
    print("Running final evaluation on validation set...")
    final_metrics = trainer.evaluate()
    print("Final Metrics:", final_metrics)
    
    # 8. Save best model to root output dir
    trainer.save_model(output_dir)
    print(f"Saved best model to {output_dir}")

if __name__ == "__main__":
    main()
