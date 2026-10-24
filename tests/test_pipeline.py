"""Tests for feature selection and the train/test split."""

import pandas as pd
import pytest

from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
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
