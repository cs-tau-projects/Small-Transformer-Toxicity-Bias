from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os



def train_baseline(X_train, y_train, model_save_path="models/baseline_model.joblib"):
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
    return pipeline

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


