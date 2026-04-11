import os
import numpy as np
import pandas as pd
from src.evaluator import evaluate_bias
from src.model.baseline import train_baseline
from src.model.naive_baseline import MajorityVoteClassifier
from src.steps.utils import load_saved_data

def run_baseline_step(data_dir, results_dir):
    train_ds, test_ds, identity_columns = load_saved_data(data_dir)

    print("\n--- Training Machine Learning Baseline (TF-IDF + LogReg) ---")
    # Clean X to ensure no None values (causes sklearn to crash)
    X_train = [str(t) if t is not None else "" for t in train_ds["comment_text"]]
    y_train = train_ds["is_toxic"]

    X_test = [str(t) if t is not None else "" for t in test_ds["comment_text"]]
    y_test = test_ds["is_toxic"]

    # Extract identity matrix for evaluation
    identities_test = [test_ds[col] for col in identity_columns]
    identity_matrix_test = np.array(identities_test).T

    # 1. Machine Learning Baseline
    pipeline_path = os.path.join(results_dir, "baseline_pipeline.joblib")
    pipeline = train_baseline(X_train, y_train, model_save_path=pipeline_path)

    print("Evaluating ML Baseline...")
    y_pred_probs_ml = pipeline.predict_proba(X_test)[:, 1]

    metrics_ml_df = evaluate_bias(
        y_true=np.array(y_test),
        y_pred_probs=y_pred_probs_ml,
        identity_matrix=identity_matrix_test,
        identity_columns=identity_columns,
        threshold=0.5,
    )

    ml_out_path = os.path.join(results_dir, "baseline_metrics.csv")
    metrics_ml_df.to_csv(ml_out_path, index=False)
    
    preds_ml_df = pd.DataFrame({'comment_text': test_ds['comment_text'], 'toxicity_score': y_pred_probs_ml})
    preds_ml_out_path = os.path.join(results_dir, "preds_Baseline.csv")
    preds_ml_df.to_csv(preds_ml_out_path, index=False)
    
    # 2. Naive Baseline (Majority Vote)
    print("\n--- Training Naive Baseline (Majority Vote) ---")
    naive_model = MajorityVoteClassifier()
    naive_model.fit(X_train, y_train)
    
    # Save the naive model for OOD reuse
    import joblib
    naive_path = os.path.join(results_dir, "naive_baseline.joblib")
    joblib.dump(naive_model, naive_path)
    print(f"Saved Naive Baseline model to {naive_path}")
    
    print("Evaluating Naive Baseline...")
    y_pred_probs_naive = naive_model.predict_proba(X_test)[:, 1]
    
    metrics_naive_df = evaluate_bias(
        y_true=np.array(y_test),
        y_pred_probs=y_pred_probs_naive,
        identity_matrix=identity_matrix_test,
        identity_columns=identity_columns,
        threshold=0.5,
    )
    
    naive_out_path = os.path.join(results_dir, "naive_baseline_metrics.csv")
    metrics_naive_df.to_csv(naive_out_path, index=False)
    
    preds_naive_df = pd.DataFrame({'comment_text': test_ds['comment_text'], 'toxicity_score': y_pred_probs_naive})
    preds_naive_out_path = os.path.join(results_dir, "preds_Naive.csv")
    preds_naive_df.to_csv(preds_naive_out_path, index=False)

    print(f"Saved ML Baseline metrics to {ml_out_path} and predictions to {preds_ml_out_path}")
    print(f"Saved Naive Baseline metrics to {naive_out_path} and predictions to {preds_naive_out_path}")
