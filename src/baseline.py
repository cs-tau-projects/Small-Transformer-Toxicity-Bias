import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

from src.data_utils import load_jigsaw_data

def train_baseline(model_save_path="models/baseline_model.joblib"):
    print("Loading Baseline Logistic Regression training data...")
    train_ds, identity_columns = load_jigsaw_data(split='train')
    
    # We use memory-mapped lists which is efficient
    print("Extracting features from dataset...")
    X_train = train_ds['comment_text']
    y_train = train_ds['is_toxic']
    
    print("Training Logistic Regression Model with TF-IDF...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    print(f"Saving baseline model to {model_save_path}...")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(pipeline, model_save_path)
    print("Model saved successfully.")

def run_baseline(test_ds, model_load_path="models/baseline_model.joblib"):
    """
    Takes the test set and returns predictions for the evaluation agent.
    """
    print(f"Loading baseline model from {model_load_path}...")
    try:
        pipeline = joblib.load(model_load_path)
    except FileNotFoundError:
        print(f"Model not found at {model_load_path}. Please train first.")
        raise
        
    print("Extracting test data texts...")
    X_test = test_ds['comment_text']
    
    print("Predicting with Baseline Model...")
    y_pred_probs = pipeline.predict_proba(X_test)[:, 1]
    
    return y_pred_probs

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Baseline pipeline for toxicity classification")
    parser.add_argument('--train', action='store_true', help='Train the baseline model')
    args = parser.parse_args()
    
    if args.train:
        train_baseline()
    else:
        print("Specify --train to train the model. For evaluation, use run_baseline function imported via other scripts.")

if __name__ == "__main__":
    main()
