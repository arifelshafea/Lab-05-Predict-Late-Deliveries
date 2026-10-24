"""Evaluate and tune models using stratified cross-validation."""

import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from latedeliveries.models import get_candidate_models
from latedeliveries.pipeline import RANDOM_STATE, build_pipeline

# Define the five classification metrics required for model comparison.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


# Compare candidate models using only the training data.
def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Return cross-validation metric means and standard deviations."""
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = []

    # Build a complete preprocessing pipeline for each candidate model.
    for name, model in get_candidate_models().items():
        pipeline = build_pipeline(model)

        # Evaluate the model with five-fold stratified cross-validation.
        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=SCORING,
        )

        row = {"model": name}

        # Calculate the mean and standard deviation of each metric.
        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        results.append(row)

    return pd.DataFrame(results)


# Tune Balanced Logistic Regression using training data only.
def tune_balanced_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> GridSearchCV:
    """Find the best settings using five-fold stratified cross-validation."""
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Keep preprocessing inside the pipeline to prevent data leakage.
    model = get_candidate_models()["balanced_logistic_regression"]
    pipeline = build_pipeline(model)

    # Test two hyperparameters across eight combinations.
    parameter_grid = {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__fit_intercept": [True, False],
    }

    # Select the settings that give the best cross-validation ROC-AUC.
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=cv,
        refit=True,
    )

    # Fit the search using only the training data.
    search.fit(X_train, y_train)

    return search


# Calculate the company's total cost of delivery warning decisions.
def order_cost(y_true, flagged) -> int:
    """Return voucher costs plus costs of missed late deliveries."""
    actual = np.asarray(y_true)
    predicted_flags = np.asarray(flagged, dtype=bool)

    # Ensure actual outcomes and warning decisions have matching sizes.
    if actual.ndim != 1 or predicted_flags.ndim != 1:
        raise ValueError("Inputs must be one-dimensional.")

    if len(actual) != len(predicted_flags):
        raise ValueError("Inputs must have the same length.")

    # Every warning costs $4, even if the delivery is on time.
    voucher_cost = 4 * np.count_nonzero(predicted_flags)

    # Every missed late delivery costs $15.
    missed_late = (actual == 1) & ~predicted_flags
    missed_cost = 15 * np.count_nonzero(missed_late)

    return int(voucher_cost + missed_cost)


# Find the classification threshold with the lowest business cost.
def choose_threshold(y_true, probabilities) -> tuple[float, int]:
    """Return the lowest-cost threshold and its total cost."""
    thresholds = [round(i * 0.05, 2) for i in range(1, 20)]

    results = []

    # Calculate the business cost at every required threshold.
    for threshold in thresholds:
        flagged = np.asarray(probabilities) >= threshold
        cost = order_cost(y_true, flagged)
        results.append((threshold, cost))

    # Choose the lowest-cost threshold; ties use the lower threshold.
    best_threshold, best_cost = min(results, key=lambda result: result[1])

    return best_threshold, best_cost
