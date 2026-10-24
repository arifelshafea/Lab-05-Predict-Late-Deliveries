"""Define delivery features and create a reproducible train/test split."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Keep the split's random state in one constant.
RANDOM_STATE = 42

# Use 75% of the records for training and 25% for testing.
TEST_SIZE = 0.25

# Numeric features available when the order is placed.
NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
]

# Categorical features available when the order is placed.
CATEGORICAL_FEATURES = [
    "weather",
    "day_type",
]


# Split the data into training and testing sets.
def split_orders(
    orders: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Return a repeatable, stratified 75/25 train/test split."""

    # Select only the approved features.
    features = orders[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

    # Separate the target from the predictive features.
    target = orders["late"]

    # Preserve the proportion of late deliveries in both sets.
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


# Build preprocessing and prediction steps for any scikit-learn model.
def build_pipeline(model):
    """Return a pipeline that preprocesses features and fits the model."""

    # Fill missing numeric values and standardize numeric features.
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # Apply the appropriate preprocessing to each feature group.
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    # Combine preprocessing and the chosen machine-learning model.
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )
