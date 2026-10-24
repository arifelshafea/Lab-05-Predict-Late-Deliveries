"""Tests for feature selection and the train/test split."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_pipeline,
    split_orders,
)


# Verify that only approved features are used.
def test_approved_features(orders):
    """Check that the split contains exactly the approved columns."""
    X_train, X_test, _, _ = split_orders(orders)

    expected = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    assert X_train.columns.tolist() == expected
    assert X_test.columns.tolist() == expected


# Check each excluded column using pytest parametrization.
@pytest.mark.parametrize(
    "excluded_column",
    [
        "order_id",
        "vehicle",
        "prep_minutes",
        "courier_months",
        "customer_rating",
        "late",
    ],
)
def test_excluded_features(orders, excluded_column):
    """Ensure unavailable or inappropriate columns are excluded."""
    X_train, X_test, _, _ = split_orders(orders)

    assert excluded_column not in X_train.columns
    assert excluded_column not in X_test.columns


# Verify the required 75/25 split.
def test_split_sizes(orders):
    """Check training and testing sizes and nonoverlapping rows."""
    X_train, X_test, y_train, y_test = split_orders(orders)

    assert len(X_train) == 2250
    assert len(X_test) == 750
    assert len(y_train) == 2250
    assert len(y_test) == 750

    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)
    assert X_train.index.intersection(X_test.index).empty


# Verify that stratification preserves the late-delivery rate.
def test_stratified_split(orders):
    """Check that the training late rate matches the original rate."""
    _, _, y_train, _ = split_orders(orders)

    original_rate = orders["late"].mean()
    training_rate = y_train.mean()

    assert abs(original_rate - training_rate) < 0.01


# Verify that the fixed seed produces repeatable results.
def test_repeatable_split(orders):
    """Check that repeating the split gives identical training data."""
    X_train_1, _, y_train_1, _ = split_orders(orders)
    X_train_2, _, y_train_2, _ = split_orders(orders)

    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)


# Verify predictions work with missing values and unseen categories.
def test_pipeline_handles_unusual_inputs(orders):
    """Check predictions with missing and unfamiliar input values."""
    X_train, _, y_train, _ = split_orders(orders)

    # Train only on the training portion of the dataset.
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    # Prepare a challenging example using a training record.
    sample = X_train.iloc[[0]].copy()
    sample["distance_km"] = float("nan")
    sample["weather"] = "hail"
    sample["prep_minutes"] = float("nan")

    # Confirm the pipeline produces a valid prediction.
    prediction = pipeline.predict(sample)
    probabilities = pipeline.predict_proba(sample)

    assert len(prediction) == 1
    assert prediction[0] in (0, 1)
    assert probabilities.shape == (1, 2)
    assert 0 <= probabilities[0, 1] <= 1
    assert probabilities[0].sum() == pytest.approx(1.0)


# Verify the numeric imputer learns the training-data median.
def test_numeric_imputer_uses_training_median(orders):
    """Check that imputation uses the training data only."""
    X_train, _, y_train, _ = split_orders(orders)

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    # Access the fitted numeric imputer inside the pipeline.
    numeric_pipeline = pipeline.named_steps["preprocess"].named_transformers_["numeric"]
    imputer = numeric_pipeline.named_steps["imputer"]

    # Compare the learned distance median with the training median.
    distance_index = NUMERIC_FEATURES.index("distance_km")

    assert imputer.statistics_[distance_index] == pytest.approx(
        X_train["distance_km"].median()
    )
