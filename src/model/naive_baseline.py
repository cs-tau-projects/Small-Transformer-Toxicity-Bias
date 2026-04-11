import numpy as np

class MajorityVoteClassifier:
    """
    A naive baseline classifier that always predicts the majority class 
    observed during training.
    """
    def __init__(self):
        self.majority_class = 0
        self.majority_prob = 0.0

    def fit(self, X, y):
        """
        Find the majority class in the labels.
        X is ignored as this is a naive baseline.
        """
        classes, counts = np.unique(y, return_counts=True)
        self.majority_class = classes[np.argmax(counts)]
        # For simplicity in prediction_probs, if the majority class is 1, 
        # we predict 1.0 probability for toxic. If it's 0, we predict 0.0.
        self.majority_prob = float(self.majority_class)
        print(f"Naive Baseline: Majority class is {self.majority_class}")

    def predict_proba(self, X):
        """
        Returns an array of probabilities for the positive class (1).
        Since this is a majority vote, it's always the same value.
        """
        n_samples = len(X)
        # We return a 2D array [P(0), P(1)] to match sklearn's predict_proba
        # But we mostly use the second column [:, 1] for AUC
        probs = np.zeros((n_samples, 2))
        if self.majority_class == 1:
            probs[:, 1] = 1.0
            probs[:, 0] = 0.0
        else:
            probs[:, 1] = 0.0
            probs[:, 0] = 1.0
        return probs
