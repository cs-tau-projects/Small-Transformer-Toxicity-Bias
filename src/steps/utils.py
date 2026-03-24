import os
import json
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from src.evaluator import evaluate_bias

def load_saved_data(data_dir):
    """Helper to load train/eval datasets and identity columns."""
    print(f"Loading cached datasets from {data_dir}...")
    baseline_train_ds = load_from_disk(os.path.join(data_dir, "baseline_train"))
    eval_ds = load_from_disk(os.path.join(data_dir, "eval"))
    
    with open(os.path.join(data_dir, "identity_columns.json"), "r") as f:
        identity_columns = json.load(f)
        
    return baseline_train_ds, eval_ds, identity_columns

def get_transformer_predictions(model, tokenizer, dataset, device, batch_size=32):
    """Generate predictions for a Transformer model on the dataset."""
    import torch
    
    # Note: This mutates the model in-place by moving it to the specified device.
    model.eval()
    model.to(device)
    all_probs = []
    
    texts = dataset['comment_text']
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing"):
            batch_texts = texts[i:i+batch_size]
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
    y_val = val_ds['is_toxic']
    
    identities_val = [val_ds[col] for col in identity_columns]
    identity_matrix_val = np.array(identities_val).T
    
    y_pred_probs = get_transformer_predictions(model, tokenizer, val_ds, device)
    
    metrics_df = evaluate_bias(
        y_true=np.array(y_val),
        y_pred_probs=y_pred_probs,
        identity_matrix=identity_matrix_val,
        identity_columns=identity_columns,
        threshold=0.5
    )
    return metrics_df, y_pred_probs
