import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch
from datasets import load_dataset
from tqdm import tqdm

def compute_subgroup_auc(y_true, y_pred, subgroup_mask):
    """
    Computes AUC restricted to the subgroup.
    Subgroup mask is a boolean array indicating if the example belongs to the subgroup.
    """
    subgroup_y_true = y_true[subgroup_mask]
    subgroup_y_pred = y_pred[subgroup_mask]
    
    # Need at least one positive and one negative example to compute AUC
    if len(np.unique(subgroup_y_true)) < 2:
        return np.nan
        
    return roc_auc_score(subgroup_y_true, subgroup_y_pred)

def compute_fnr(y_true, y_pred_binary):
    """Computes False Negative Rate."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return fn / (fn + tp) if (fn + tp) > 0 else np.nan

def compute_fnr_gap(y_true, y_pred_binary, subgroup_mask):
    """
    Computes the False Negative Rate gap = FNR_subgroup - FNR_overall.
    Positive gap means the model misses toxicity more often for this subgroup.
    """
    overall_fnr = compute_fnr(y_true, y_pred_binary)
    
    subgroup_y_true = y_true[subgroup_mask]
    subgroup_y_pred_binary = y_pred_binary[subgroup_mask]
    subgroup_fnr = compute_fnr(subgroup_y_true, subgroup_y_pred_binary)
    
    if np.isnan(overall_fnr) or np.isnan(subgroup_fnr):
        return np.nan
        
    return subgroup_fnr - overall_fnr

def compute_pinned_auc(y_true, y_pred_probs, subgroup_mask, random_state=42):
    """
    Computes Pinned AUC as defined by Dixon et al. (2018).
    Creates a balanced dataset containing:
    - The full subgroup distribution.
    - An equally-sized random sample from the background (overall) distribution.
    """
    subgroup_indices = np.where(subgroup_mask)[0]
    background_indices = np.where(~subgroup_mask)[0]
    
    # If the subgroup is empty, or the background is empty, we can't compute
    if len(subgroup_indices) == 0 or len(background_indices) == 0:
        return np.nan
        
    # Sample background to match subgroup size
    np.random.seed(random_state)
    sample_size = min(len(subgroup_indices), len(background_indices))
    sampled_background_indices = np.random.choice(background_indices, size=sample_size, replace=False)
    
    # Optional: Since equation (3) says |s(Dt)| = |s(D)|, we can also sub-sample the subgroup 
    # to match the background if the subgroup is larger. Using `sample_size` ensures balance.
    sampled_subgroup_indices = np.random.choice(subgroup_indices, size=sample_size, replace=False)
    
    pinned_indices = np.concatenate([sampled_subgroup_indices, sampled_background_indices])
    pinned_y_true = y_true[pinned_indices]
    pinned_y_pred_probs = y_pred_probs[pinned_indices]
    
    if len(np.unique(pinned_y_true)) < 2:
        return np.nan
        
    return roc_auc_score(pinned_y_true, pinned_y_pred_probs)

def evaluate_bias(y_true, y_pred_probs, identity_matrix, identity_columns, threshold=0.5):
    """
    Evaluates predictions against the ground truth and identity annotations.
    Computes Overall AUC, Subgroup AUC, Subgroup FNR Gap, and Pinned AUC.
    
    Args:
        y_true: np.array of shape (N,) with ground truth binary labels
        y_pred_probs: np.array of shape (N,) with predicted probabilities for the positive class
        identity_matrix: np.array of shape (N, num_identities) with continuous identity scores or binary flags
        identity_columns: list of strings with the names of the identities
        threshold: float, threshold to binarize predictions for FNR computation (default 0.5)
        
    Returns:
        pd.DataFrame: Table with Overall AUC, Subgroup AUC, FNR gaps, and Pinned AUC for each identity.
    """
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    
    overall_auc = roc_auc_score(y_true, y_pred_probs)
    overall_fnr = compute_fnr(y_true, y_pred_binary)
    
    metrics = []
    
    for i, col in enumerate(identity_columns):
        # The dataset logic leaves identity values continuous.
        # We consider a subgroup "active" for an example if the score > 0.5
        subgroup_scores = identity_matrix[:, i]
        subgroup_mask = subgroup_scores > 0.5
        
        # Calculate metrics if there are any examples in the subgroup
        if subgroup_mask.sum() > 0:
            subgroup_auc = compute_subgroup_auc(y_true, y_pred_probs, subgroup_mask)
            fnr_gap = compute_fnr_gap(y_true, y_pred_binary, subgroup_mask)
            pinned_auc = compute_pinned_auc(y_true, y_pred_probs, subgroup_mask)
            num_examples = subgroup_mask.sum()
        else:
            subgroup_auc = np.nan
            fnr_gap = np.nan
            pinned_auc = np.nan
            num_examples = 0
            
        metrics.append({
            "Identity": col,
            "Total Examples": num_examples,
            "1. Overall AUC": overall_auc,
            "2. Overall FNR": overall_fnr,
            "3. Subgroup AUC": subgroup_auc,
            "4. Subgroup FNR Gap": fnr_gap,
            "5. Pinned AUC": pinned_auc
        })
        
    df = pd.DataFrame(metrics)
    return df

# Alias to maintain backwards compatibility while adopting the requested `evaluate_bias` name
evaluate_models_metrics = evaluate_bias

def evaluate_sorry_bench(model, tokenizer, device, num_samples=200, batch_size=32):
    """
    Evaluates a fine-tuned model on a subset of the SORRY-bench safety prompts.
    Since SORRY-bench contains prompts designed to test safety evaluation in LLMs, we can
    use the texts to see how often our simple toxicity classifier flags them as toxic.
    High toxicity rate on these prompts might indicate generalization to diverse safety issues.
    
    Args:
        model: Hugging Face model
        tokenizer: Hugging Face tokenizer
        device: torch.device
        num_samples: int, number of samples to evaluate on
        batch_size: int
        
    Returns:
        float: The proportion of prompts classified as toxic.
    """
    print(f"Loading SORRY-bench dataset ({num_samples} samples)...")
    try:
        ds = load_dataset("sorry-bench/sorry-bench-202406", split="test")
        # Subsample to keep evaluation quick
        ds = ds.shuffle(seed=42).select(range(min(num_samples, len(ds))))
    except Exception as e:
        print(f"Failed to load SORRY-bench: {e}")
        return np.nan
        
    texts = [item['prompt'] for item in ds]
    model.eval()
    
    all_preds = []
    
    print("Evaluating on SORRY-bench...")
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Assuming toxicity is class 1
            if logits.shape[1] == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1]
            else:
                probs = torch.sigmoid(logits)[:, 0]
                
            preds = (probs > 0.5).int().cpu().numpy()
            all_preds.extend(preds)
            
    toxicity_rate = np.mean(all_preds)
    return toxicity_rate
