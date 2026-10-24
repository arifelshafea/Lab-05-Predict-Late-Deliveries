"""Test saving and reloading the complete prediction pipeline."""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from latedeliveries.data import make_orders
from latedeliveries.pipeline import build_pipeline, split_orders


# Verify that saving and reloading preserves model predictions.
def test_saved_model_predictions_match(tmp_path):
    """Reload a saved pipeline and compare its predictions."""
    # Use training data only; do not evaluate the reserved test set.
    X_train, _, y_train, _ = split_orders(make_orders())

    # Fit the complete preprocessing and classification pipeline.
    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    # Record predictions before saving the model.
    sample = X_train.iloc[:5]
    original_predictions = pipeline.predict(sample)
    original_probabilities = pipeline.predict_proba(sample)

    # Save the pipeline in pytest's temporary directory.
    model_path = tmp_path / "late_orders.joblib"
    joblib.dump(pipeline, model_path)

    assert model_path.is_file()

    # Reload the model created by this test.
    loaded_pipeline = joblib.load(model_path)

    # Predictions and probabilities must remain unchanged.
    np.testing.assert_array_equal(
        loaded_pipeline.predict(sample),
        original_predictions,
    )
    np.testing.assert_allclose(
        loaded_pipeline.predict_proba(sample),
        original_probabilities,
    )
