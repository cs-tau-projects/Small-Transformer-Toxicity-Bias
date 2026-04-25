"""
Sanity Check / Overfitting Experiment
--------------------------------------
Trains each finetuned model on a tiny subset (~100 samples) for many epochs
and verifies that training loss drops to near zero and accuracy exceeds 99%.

This confirms correct implementation (no data leaks, correct loss, proper
gradient flow) and satisfies the assignment's methodology requirement:
  "make sure you are able to overfit on a small sample of the data"

Usage:
  python -m src.sanity_check --output_dir ./outputs
  python -m src.sanity_check --output_dir ./outputs --n_samples 200 --epochs 20
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_callback import PrinterCallback

from src.data.data_utils import get_hf_token
from src.data.dataset import (
    JigsawDataset,
    download_and_prep_jigsaw,
    tokenize_jigsaw_dataset,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sanity_check")


MODELS = [
    "distilbert-base-uncased",
    "distilroberta-base",
    "google/bert_uncased_L-4_H-512_A-8",
]


def compute_accuracy(eval_pred):
    """Simple accuracy + AUC for the sanity check — no bias metrics needed."""
    logits, labels = eval_pred
    if logits.shape[1] == 2:
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        probs_pos = probs[:, 1]
    else:
        probs_pos = 1 / (1 + np.exp(-logits[:, 0]))

    preds = (probs_pos >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs_pos)
    except ValueError:
        auc = float("nan")
    return {"accuracy": acc, "roc_auc": auc}


def run_sanity_check(output_dir, n_samples=100, epochs=20, seed=42):
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    cache_dir = os.path.join(output_dir, ".cache")
    data_dir = os.path.join(output_dir, "data")
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    hf_token = get_hf_token()

    # --- Load data --------------------------------------------------------
    # Prefer pre-saved splits if available (consistent with main pipeline)
    train_path = os.path.join(data_dir, "train")
    id_cols_path = os.path.join(data_dir, "identity_columns.json")

    if os.path.exists(train_path) and os.path.exists(id_cols_path):
        logger.info("Loading pre-saved training split from %s", data_dir)
        train_hf = load_from_disk(train_path)
        with open(id_cols_path) as f:
            identity_columns = json.load(f)
    else:
        logger.info("No pre-saved splits found — downloading Jigsaw dataset")
        train_hf, identity_columns = download_and_prep_jigsaw("train", cache_dir=cache_dir)
        train_hf = train_hf.shuffle(seed=seed)

    # Take a tiny subset
    subset = train_hf.select(range(min(n_samples, len(train_hf))))
    logger.info("Sanity check subset: %d samples", len(subset))

    # --- Run for each model -----------------------------------------------
    results = []
    all_passed = True

    for model_name in MODELS:
        logger.info("=" * 60)
        logger.info("Sanity check: %s", model_name)
        logger.info("=" * 60)

        # Tokenize
        tokenized = tokenize_jigsaw_dataset(subset, model_name, cache_dir=cache_dir)
        dataset = JigsawDataset(tokenized, identity_columns)

        # Load fresh model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, cache_dir=cache_dir, token=hf_token,
        )

        safe_name = model_name.replace("/", "_")
        sanity_output = os.path.join(output_dir, f"sanity_check_{safe_name}")

        training_args = TrainingArguments(
            output_dir=sanity_output,
            num_train_epochs=epochs,
            per_device_train_batch_size=min(16, n_samples),
            per_device_eval_batch_size=min(16, n_samples),
            learning_rate=2e-5,
            warmup_ratio=0.0,  # No warmup — we want fast convergence
            eval_strategy="epoch",
            save_strategy="no",  # Don't save checkpoints for sanity check
            seed=seed,
            fp16=torch.cuda.is_available(),
            logging_steps=1,  # Log every step so we can see loss drop
            log_level="info",
            disable_tqdm=False,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=dataset,  # Evaluate on the SAME data — we want overfitting
            compute_metrics=compute_accuracy,
        )
        trainer.remove_callback(PrinterCallback)

        # Train
        train_result = trainer.train()

        # Final evaluation on the training subset itself
        eval_metrics = trainer.evaluate()

        final_loss = train_result.training_loss
        final_acc = eval_metrics["eval_accuracy"]
        final_auc = eval_metrics.get("eval_roc_auc", float("nan"))
        passed = final_acc >= 0.99

        if not passed:
            all_passed = False

        status = "PASS" if passed else "FAIL"
        logger.info("-" * 60)
        logger.info(
            "[%s] %s — loss: %.4f | accuracy: %.4f | AUC: %.4f",
            status, model_name, final_loss, final_acc, final_auc,
        )
        logger.info("-" * 60)

        results.append({
            "model": model_name,
            "n_samples": n_samples,
            "epochs": epochs,
            "final_train_loss": round(final_loss, 6),
            "final_accuracy": round(final_acc, 4),
            "final_auc": round(final_auc, 4) if not np.isnan(final_auc) else None,
            "passed": passed,
        })

        # Clean up GPU memory between models
        del model, trainer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Summary ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SANITY CHECK SUMMARY")
    logger.info("=" * 60)
    for r in results:
        tag = "PASS" if r["passed"] else "FAIL"
        logger.info(
            "  [%s] %-45s acc=%.4f  loss=%.6f",
            tag, r["model"], r["final_accuracy"], r["final_train_loss"],
        )

    # Save results
    results_path = os.path.join(results_dir, "sanity_check_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    if all_passed:
        logger.info("All models passed the sanity check (accuracy >= 99%%).")
    else:
        logger.error("Some models FAILED the sanity check. Investigate before proceeding.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Sanity check: overfit on a small data subset.")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Base output directory (same as main pipeline).")
    parser.add_argument("--n_samples", type=int, default=100,
                        help="Number of training samples for the overfit test.")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs (enough to overfit).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    args = parser.parse_args()

    run_sanity_check(
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        epochs=args.epochs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
