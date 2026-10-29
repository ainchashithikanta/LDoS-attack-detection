"""
Module: random_forest_model.py
Purpose: Implements the Random Forest ensemble classifier for LDoS detection,
         providing higher robustness, variance reduction, and feature importance analysis.

Background for Students:
------------------------
A Random Forest builds an ensemble of multiple decorrelated decision trees, each trained
on a bootstrap sample of the training data (Bagging) and a random subset of features.
- Voting mechanism: The forest averages the probabilistic predictions of individual trees,
  significantly reducing variance and false alarms.
- Hyperparameters: Uses n_estimators=100 trees and max_depth=5 to ensure high generalization
  without overfitting.
"""

from typing import Tuple
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config


def build_random_forest_pipeline(
    n_estimators: int = config.RF_N_ESTIMATORS,
    max_depth: int = config.RF_MAX_DEPTH,
    random_state: int = config.RANDOM_STATE,
    use_scaler: bool = True
) -> Pipeline:
    """
    Constructs a Scikit-learn Pipeline with StandardScaler and RandomForestClassifier.

    Parameters:
    -----------
    n_estimators : int
        Number of decision trees in forest (default: 100).
    max_depth : int
        Maximum depth of each tree (default: 5).
    random_state : int
        Seed for reproducibility (default: 42).
    use_scaler : bool
        Whether to standardize features prior to training (default: True).

    Returns:
    --------
    pipeline : Pipeline
        Unfitted Scikit-learn pipeline.
    """
    steps = []
    if use_scaler:
        steps.append(("scaler", StandardScaler()))

    steps.append((
        "classifier",
        RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            criterion="gini",
            class_weight="balanced"
        )
    ))

    return Pipeline(steps)


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = config.RF_N_ESTIMATORS,
    max_depth: int = config.RF_MAX_DEPTH,
    random_state: int = config.RANDOM_STATE
) -> Tuple[Pipeline, float]:
    """
    Trains the Random Forest pipeline strictly on training data.

    Parameters:
    -----------
    X_train : np.ndarray
        Training feature matrix (n_samples, n_features).
    y_train : np.ndarray
        Training labels.
    n_estimators : int
        Number of trees.
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
    pipeline = build_random_forest_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    return pipeline, train_time


def predict_random_forest(
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

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(X_test)[:, 1]
    else:
        probabilities = predictions.astype(float)

    return predictions, probabilities, inference_time


if __name__ == "__main__":
    print("Testing random_forest_model.py...")
    X_dummy = np.random.randn(20, 6)
    y_dummy = np.random.randint(0, 2, 20)
    pipe, t_train = train_random_forest(X_dummy, y_dummy)
    preds, probs, t_pred = predict_random_forest(pipe, X_dummy[:5])
    print("Predictions:", preds)
    print(f"Train time: {t_train*1000:.2f}ms | Inference time: {t_pred*1000:.2f}ms")
    assert len(preds) == 5
    print("Random Forest module successfully verified!")
