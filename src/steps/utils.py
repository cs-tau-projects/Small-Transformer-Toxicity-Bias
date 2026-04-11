import json
import logging
import os
from datetime import datetime

import numpy as np
from datasets import load_from_disk
from rich.logging import RichHandler
from tqdm import tqdm

from src.evaluator import evaluate_bias


def setup_logging(output_dir):
    """Sets up centralized logging for the pipeline."""
    log_dir = os.path.join(output_dir)
    os.makedirs(log_dir, exist_ok=True)

    # Include timestamp in the log filename to unique runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True), logging.FileHandler(log_file, mode="a")],
    )
    return logging.getLogger("pipeline")


def load_saved_data(data_dir):
    """
    Helper to load datasets and identity columns.
    By default, returns (Train, Test) to downstream evaluation steps.
    """
    print(f"Loading cached datasets from {data_dir}...")
    
    # Prioritize new 80/10/10 split names
    train_path = os.path.join(data_dir, "train")
    val_path = os.path.join(data_dir, "val")
    test_path = os.path.join(data_dir, "test")
    
    if os.path.exists(train_path) and os.path.exists(test_path):
        train_ds = load_from_disk(train_path)
        test_ds = load_from_disk(test_path)
    else:
        # Fallback to 90/10 names
        train_ds = load_from_disk(os.path.join(data_dir, "baseline_train"))
        test_ds = load_from_disk(os.path.join(data_dir, "eval"))

    with open(os.path.join(data_dir, "identity_columns.json"), "r") as f:
        identity_columns = json.load(f)

    return train_ds, test_ds, identity_columns


def get_transformer_predictions(model, tokenizer, dataset, device, batch_size=32):
    """Generate predictions for a Transformer model on the dataset."""
    import torch

    # Note: This mutates the model in-place by moving it to the specified device.
    model.eval()
    model.to(device)
    all_probs = []

    texts = dataset["comment_text"]

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing"):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            logits = outputs.logits

            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(logits)[:, 0]

            all_probs.extend(probs.cpu().numpy())

    return np.array(all_probs)


def eval_transformer(model_desc, model, tokenizer, val_ds, identity_columns, device):
    """Evaluates a Transformer model given the validation dataset."""
    print(f"\n--- Evaluating {model_desc} ---")
    y_val = val_ds["is_toxic"]

    identities_val = [val_ds[col] for col in identity_columns]
    identity_matrix_val = np.array(identities_val).T

    y_pred_probs = get_transformer_predictions(model, tokenizer, val_ds, device)

    metrics_df = evaluate_bias(
        y_true=np.array(y_val),
        y_pred_probs=y_pred_probs,
        identity_matrix=identity_matrix_val,
        identity_columns=identity_columns,
        threshold=0.5,
    )
    return metrics_df, y_pred_probs
