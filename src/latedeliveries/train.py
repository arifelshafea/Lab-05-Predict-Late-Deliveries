"""Train, evaluate, and save the late-delivery prediction pipeline."""

import json
from pathlib import Path

import joblib
import sklearn
from sklearn.base import clone
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from latedeliveries.data import make_orders
from latedeliveries.evaluation import (
    choose_threshold,
    compare_models,
    order_cost,
    tune_balanced_logistic,
)
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    split_orders,
)

# Locate the project root so output paths work from any directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Reproduce the complete training and evaluation workflow.
def main() -> None:
    """Train the model, evaluate it once, and save required outputs."""

    # Create the required output directories.
    reports_dir = PROJECT_ROOT / "reports"
    models_dir = PROJECT_ROOT / "models"
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Generate orders and reserve the test set for final evaluation.
    orders = make_orders()
    X_train, X_test, y_train, y_test = split_orders(orders)

    # Compare all models using training data only.
    cv_results = compare_models(X_train, y_train)
    cv_results.to_csv(reports_dir / "cv_results.csv", index=False)

    # Find the best hyperparameters using training data only.
    search = tune_balanced_logistic(X_train, y_train)

    # Generate out-of-fold training probabilities for threshold selection.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probabilities = cross_val_predict(
        search.best_estimator_,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    # Select the threshold that minimizes training-stage business cost.
    threshold, training_cost = choose_threshold(
        y_train,
        probabilities,
    )

    # Fit a fresh final pipeline using the entire training set.
    final_pipeline = clone(search.best_estimator_)
    final_pipeline.fit(X_train, y_train)

    # Evaluate the reserved test set once, after all decisions are fixed.
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]
    flagged = test_probabilities >= threshold

    # Calculate the required classification metrics.
    metrics = {
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(
            y_test,
            flagged,
            labels=[0, 1],
        ).tolist(),
        "precision": float(precision_score(y_test, flagged, zero_division=0)),
        "recall": float(recall_score(y_test, flagged)),
        "f1": float(f1_score(y_test, flagged)),
        "roc_auc": float(roc_auc_score(y_test, test_probabilities)),
        "costs": {
            "flag_nobody": order_cost(
                y_test,
                [False] * len(y_test),
            ),
            "flag_everybody": order_cost(
                y_test,
                [True] * len(y_test),
            ),
            "model": order_cost(y_test, flagged),
        },
    }

    # Save final test metrics and business costs.
    with (reports_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    # Save the complete fitted preprocessing and prediction pipeline.
    joblib.dump(final_pipeline, models_dir / "late_orders.joblib")

    # Record information needed to understand and reload the model.
    model_info = {
        "scikit_learn_version": sklearn.__version__,
        "random_state": RANDOM_STATE,
        "threshold": float(threshold),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_settings": search.best_params_,
        "training_cv_cost": int(training_cost),
        "test_results": metrics,
    }

    with (models_dir / "model_info.json").open("w", encoding="utf-8") as file:
        json.dump(model_info, file, indent=2)

    # Display a short summary of the completed training run.
    print("Training and evaluation completed.")
    print(f"Best settings: {search.best_params_}")
    print(f"Selected threshold: {threshold:.2f}")
    print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Test model cost: ${metrics['costs']['model']}")
    print("Model and reports saved successfully.")


# Allow the professor to run the project with python -m.
if __name__ == "__main__":
    main()
