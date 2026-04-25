import logging
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger("pipeline")


def train_baseline(X_train, y_train, model_save_path="models/baseline_model.joblib"):
    logger.info("Training Logistic Regression Model with TF-IDF (with CV over C)...")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=10000, stop_words="english")),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        ]
    )

    param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}

    from sklearn.model_selection import GridSearchCV

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    best_C = grid_search.best_params_["clf__C"]
    logger.info(f"Best C={best_C} (CV ROC-AUC: {grid_search.best_score_:.4f})")

    best_pipeline = grid_search.best_estimator_

    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    joblib.dump(best_pipeline, model_save_path)
    logger.info(f"Saved baseline model to {model_save_path}")
    return best_pipeline

