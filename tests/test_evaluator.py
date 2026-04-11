import numpy as np
import pytest
from src.evaluator import compute_fnr, compute_fpr, compute_subgroup_auc, evaluate_bias
import pandas as pd

def test_compute_fnr_all_positives():
    # Only positives in ground truth
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([1, 1, 0, 0]) # 2 TP, 2 FN -> fnr = 2/(2+2) = 0.5
    assert compute_fnr(y_true, y_pred) == 0.5

def test_compute_fnr_all_negatives():
    # Only negatives in ground truth -> FNR should be nan because there are no positives
    y_true = np.array([0, 0, 0])
    y_pred = np.array([1, 1, 0])
    assert np.isnan(compute_fnr(y_true, y_pred))

def test_compute_fpr_all_negatives():
    # Only negatives in ground truth
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0]) # 2 FP, 2 TN -> fpr = 2/(2+2) = 0.5
    assert compute_fpr(y_true, y_pred) == 0.5

def test_compute_fpr_all_positives():
    # Only positives in ground truth -> FPR should be nan because there are no negatives
    y_true = np.array([1, 1, 1])
    y_pred = np.array([1, 1, 0])
    assert np.isnan(compute_fpr(y_true, y_pred))

def test_compute_subgroup_auc():
    # Proper
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2])
    mask = np.array([True, True, True, True])
    # AUC should be 1.0
    assert compute_subgroup_auc(y_true, y_pred, mask) == 1.0
    
    # Missing classes in subgroup
    mask_only_pos = np.array([True, False, True, False])
    assert np.isnan(compute_subgroup_auc(y_true, y_pred, mask_only_pos))

def test_evaluate_bias():
    y_true = np.array([1, 0, 1, 0])
    y_pred_probs = np.array([0.9, 0.1, 0.4, 0.6])
    
    # Let's say identity 0 covers the first two examples, identity 1 covers last two.
    identity_matrix = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    
    identity_columns = ["ident0", "ident1"]
    
    df = evaluate_bias(y_true, y_pred_probs, identity_matrix, identity_columns)
    
    assert len(df) == 2
    row0 = df[df["Identity"] == "ident0"].iloc[0]
    row1 = df[df["Identity"] == "ident1"].iloc[0]
    
    # ident0: y_true=[1, 0], y_pred=[0.9, 0.1]
    # pred_binary = [1, 0]
    # FNR = 0.0, FPR = 0.0, AUC = 1.0
    assert row0["5. Subgroup FNR"] == 0.0
    assert row0["6. Subgroup FPR"] == 0.0
    assert row0["4. Subgroup AUC"] == 1.0
    assert row0["Total Examples"] == 2
    
    # ident1: y_true=[1, 0], y_pred=[0.4, 0.6]
    # pred_binary = [0, 1]
    # Here, TP=0, FN=1 -> FNR=1.0
    # FP=1, TN=0 -> FPR=1.0
    # AUC=0.0
    assert row1["5. Subgroup FNR"] == 1.0
    assert row1["6. Subgroup FPR"] == 1.0
    assert row1["4. Subgroup AUC"] == 0.0
    assert row1["Total Examples"] == 2
