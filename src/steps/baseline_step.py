import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.evaluator import evaluate_bias
from src.steps.utils import load_saved_data

def run_baseline_step(data_dir, results_dir):
    baseline_train_ds, eval_ds, identity_columns = load_saved_data(data_dir)
    
    print("\n--- Training Baseline (TF-IDF + LogReg) ---")
    # Clean X to ensure no None values (causes sklearn to crash)
    X_train = [str(t) if t is not None else "" for t in baseline_train_ds['comment_text']]
    y_train = baseline_train_ds['is_toxic']
    
    X_val = [str(t) if t is not None else "" for t in eval_ds['comment_text']]
    y_val = eval_ds['is_toxic']
    
    # Extract identity matrix for evaluation
    identities_val = [eval_ds[col] for col in identity_columns]
    identity_matrix_val = np.array(identities_val).T
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print("Evaluating Baseline...")
    y_pred_probs = pipeline.predict_proba(X_val)[:, 1]
    
    metrics_df = evaluate_bias(
        y_true=np.array(y_val),
        y_pred_probs=y_pred_probs,
        identity_matrix=identity_matrix_val,
        identity_columns=identity_columns,
        threshold=0.5
    )
    
    out_path = os.path.join(results_dir, "baseline_metrics.csv")
    metrics_df.to_csv(out_path, index=False)
    print(f"Saved Baseline metrics to {out_path}")
