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

def compute_fpr(y_true, y_pred_binary):
    """Computes False Positive Rate."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else np.nan


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
        pd.DataFrame: Table with Overall AUC, Overall FNR, Overall FPR, Subgroup AUC, Subgroup FNR, and Subgroup FPR for each identity.
    """
    y_pred_binary = (y_pred_probs >= threshold).astype(int)
    
    overall_auc = roc_auc_score(y_true, y_pred_probs)
    overall_fnr = compute_fnr(y_true, y_pred_binary)
    overall_fpr = compute_fpr(y_true, y_pred_binary)
    
    metrics = []
    
    for i, col in enumerate(identity_columns):
        # The dataset logic leaves identity values continuous.
        # We consider a subgroup "active" for an example if the score > 0.5
        subgroup_scores = identity_matrix[:, i]
        subgroup_mask = subgroup_scores > 0.5
        
        # Calculate metrics if there are any examples in the subgroup
        if subgroup_mask.sum() > 0:
            subgroup_auc = compute_subgroup_auc(y_true, y_pred_probs, subgroup_mask)
            subgroup_fnr = compute_fnr(y_true[subgroup_mask], y_pred_binary[subgroup_mask])
            subgroup_fpr = compute_fpr(y_true[subgroup_mask], y_pred_binary[subgroup_mask])
            num_examples = subgroup_mask.sum()
        else:
            subgroup_auc = np.nan
            subgroup_fnr = np.nan
            subgroup_fpr = np.nan
            num_examples = 0
            
        metrics.append({
            "Identity": col,
            "Total Examples": num_examples,
            "1. Overall AUC": overall_auc,
            "2. Overall FNR": overall_fnr,
            "3. Overall FPR": overall_fpr,
            "4. Subgroup AUC": subgroup_auc,
            "5. Subgroup FNR": subgroup_fnr,
            "6. Subgroup FPR": subgroup_fpr
        })
        
    df = pd.DataFrame(metrics)
    return df

# Alias to maintain backwards compatibility while adopting the requested `evaluate_bias` name
evaluate_models_metrics = evaluate_bias


