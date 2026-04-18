import numpy as np
import pytest
from src.evaluator import compute_fnr, compute_fpr, compute_subgroup_auc, compute_bpsn_auc, compute_bnsp_auc, evaluate_bias
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

def test_compute_bpsn_auc_perfect():
    """
    BPSN AUC: background positives (toxic, not in subgroup) vs
    subgroup negatives (non-toxic, in subgroup).
    
    Perfect case: background toxic scored higher than subgroup non-toxic.
    """
    # Examples: [bg_toxic, bg_toxic, sg_nontoxic, sg_nontoxic]
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.1, 0.2])  # toxic scored higher
    subgroup_mask = np.array([False, False, True, True])
    
    assert compute_bpsn_auc(y_true, y_pred, subgroup_mask) == 1.0

def test_compute_bpsn_auc_worst():
    """
    Worst case: subgroup non-toxic scored higher than background toxic.
    This means the model over-flags the subgroup.
    """
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.9, 0.8])  # non-toxic subgroup scored higher
    subgroup_mask = np.array([False, False, True, True])
    
    assert compute_bpsn_auc(y_true, y_pred, subgroup_mask) == 0.0

def test_compute_bpsn_auc_nan_no_background_positives():
    """Returns NaN when there are no background positives (all toxic in subgroup)."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.8, 0.1, 0.2])
    subgroup_mask = np.array([True, True, True, True])  # all in subgroup -> no background
    
    # Only subgroup negatives exist, no background positives -> only class 0 in subset
    assert np.isnan(compute_bpsn_auc(y_true, y_pred, subgroup_mask))

def test_compute_bnsp_auc_perfect():
    """
    BNSP AUC: background negatives (non-toxic, not in subgroup) vs
    subgroup positives (toxic, in subgroup).
    
    Perfect case: subgroup toxic scored higher than background non-toxic.
    """
    # Examples: [bg_nontoxic, bg_nontoxic, sg_toxic, sg_toxic]
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.2, 0.9, 0.8])  # toxic scored higher
    subgroup_mask = np.array([False, False, True, True])
    
    assert compute_bnsp_auc(y_true, y_pred, subgroup_mask) == 1.0

def test_compute_bnsp_auc_worst():
    """
    Worst case: subgroup toxic scored lower than background non-toxic.
    This means the model under-flags the subgroup.
    """
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.9, 0.8, 0.1, 0.2])  # toxic subgroup scored lower
    subgroup_mask = np.array([False, False, True, True])
    
    assert compute_bnsp_auc(y_true, y_pred, subgroup_mask) == 0.0

def test_compute_bnsp_auc_nan_no_subgroup_positives():
    """Returns NaN when there are no subgroup positives."""
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    subgroup_mask = np.array([False, False, True, True])
    
    assert np.isnan(compute_bnsp_auc(y_true, y_pred, subgroup_mask))

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
    
    # Check that new columns exist
    assert "5. BPSN AUC" in df.columns
    assert "6. BNSP AUC" in df.columns
    
    # ident0: y_true=[1, 0], y_pred=[0.9, 0.1]
    # pred_binary = [1, 0]
    # FNR = 0.0, FPR = 0.0, AUC = 1.0
    assert row0["7. Subgroup FNR"] == 0.0
    assert row0["8. Subgroup FPR"] == 0.0
    assert row0["4. Subgroup AUC"] == 1.0
    assert row0["Total Examples"] == 2
    
    # ident1: y_true=[1, 0], y_pred=[0.4, 0.6]
    # pred_binary = [0, 1]
    # Here, TP=0, FN=1 -> FNR=1.0
    # FP=1, TN=0 -> FPR=1.0
    # AUC=0.0
    assert row1["7. Subgroup FNR"] == 1.0
    assert row1["8. Subgroup FPR"] == 1.0
    assert row1["4. Subgroup AUC"] == 0.0
    assert row1["Total Examples"] == 2

def test_evaluate_bias_bpsn_bnsp_values():
    """
    Test that BPSN and BNSP AUC values are computed correctly in evaluate_bias.
    
    Setup: 8 examples, 2 identities. 
    - Identity A: examples 4-7 (indices), mixed toxic/non-toxic
    - Background (not A): examples 0-3, mixed toxic/non-toxic
    """
    y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    # Perfect model for background, perfect model for subgroup
    y_pred_probs = np.array([0.9, 0.8, 0.1, 0.2, 0.9, 0.8, 0.1, 0.2])
    
    identity_matrix = np.array([
        [0.0],  # bg
        [0.0],  # bg
        [0.0],  # bg
        [0.0],  # bg
        [1.0],  # subgroup
        [1.0],  # subgroup
        [1.0],  # subgroup
        [1.0],  # subgroup
    ])
    
    identity_columns = ["test_identity"]
    df = evaluate_bias(y_true, y_pred_probs, identity_matrix, identity_columns)
    
    row = df.iloc[0]
    # Perfect model: BPSN should be 1.0 (bg toxic > sg non-toxic)
    assert row["5. BPSN AUC"] == 1.0
    # Perfect model: BNSP should be 1.0 (sg toxic > bg non-toxic)
    assert row["6. BNSP AUC"] == 1.0
