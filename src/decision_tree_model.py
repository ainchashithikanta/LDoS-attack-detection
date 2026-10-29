"""
Module: decision_tree_model.py
Purpose: Implements the baseline Decision Tree classifier for LDoS detection,
         matching the model architecture in the original research paper.

Background for Students:
------------------------
A Decision Tree splits data based on feature thresholds that maximize class purity
(e.g. using Gini impurity or Information Gain). In LDoS detection:
- It provides high interpretability: students can easily inspect which feature
  (e.g. energy ratio or contrast) serves as the primary root decision node.
- Maximum depth is constrained to `max_depth=3` to prevent overfitting on small
  traffic windows and ensure fast real-time inference.
"""

from typing import Tuple, Dict, Any
import time
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def build_decision_tree_pipeline(
    max_depth: int = config.DT_MAX_DEPTH,
    random_state: int = config.RANDOM_STATE,
    use_scaler: bool = False
) -> Pipeline:
    """
    Constructs a Scikit-learn Pipeline containing an optional StandardScaler
    and the DecisionTreeClassifier.

    Parameters:
    -----------
    max_depth : int
        Maximum depth of tree to prevent overfitting (default: 3).
    random_state : int
        Seed for reproducibility (default: 42).
    use_scaler : bool
        Whether to standardize features prior to tree training (default: False,
        matching the original research paper where unscaled raw features are used).
    """
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append((
        "classifier",
        DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state,
            criterion="gini"
        )
    ))

    return Pipeline(steps)


def train_decision_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_depth: int = config.DT_MAX_DEPTH,
    random_state: int = config.RANDOM_STATE,
    use_scaler: bool = False
) -> Tuple[Pipeline, float]:
    """
    Trains the Decision Tree pipeline strictly on training data.

    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix (n_samples, n_features).
    y_train : np.ndarray
        Training labels.
    max_depth : int
        Max tree depth.
    random_state : int
        Random seed.

    Returns:
    --------
    pipeline : Pipeline
        Fitted Scikit-learn pipeline.
    train_time : float
        Elapsed training time in seconds.
    """
    pipeline = build_decision_tree_pipeline(
        max_depth=max_depth, random_state=random_state, use_scaler=use_scaler
    )

    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    return pipeline, train_time


def predict_decision_tree(
    pipeline: Pipeline,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Generates class predictions and class probabilities on the test set.

    Parameters:
    -----------
    pipeline : Pipeline
        Trained pipeline.
    X_test : np.ndarray
        Test feature matrix.

    Returns:
    --------
    predictions : np.ndarray
        Predicted binary labels (0 or 1).
    probabilities : np.ndarray
        Predicted class probabilities for LDoS class (1).
    inference_time : float
        Elapsed test prediction time in seconds.
    """
    start_time = time.perf_counter()
    predictions = pipeline.predict(X_test)
    inference_time = time.perf_counter() - start_time

    # Probabilities for positive class (label 1)
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]
    else:
        probabilities = predictions.astype(float)

    return predictions, probabilities, inference_time


if __name__ == "__main__":
    print("Testing decision_tree_model.py...")
    X_dummy = np.random.randn(20, 4)
    y_dummy = np.random.randint(0, 2, 20)
    pipe, t_train = train_decision_tree(X_dummy, y_dummy)
    preds, probs, t_pred = predict_decision_tree(pipe, X_dummy[:5])
    print("Predictions:", preds)
    print(f"Train time: {t_train*1000:.2f}ms | Inference time: {t_pred*1000:.2f}ms")
    assert len(preds) == 5
    print("Decision Tree module successfully verified!")
