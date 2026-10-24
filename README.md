# Late Deliveries

This project predicts whether a QuickBite food delivery order will arrive late using information available when the order is placed. The goal is to reduce business costs by identifying likely delays and offering customers a $4 voucher.

The project uses a machine learning pipeline, stratified cross-validation, hyperparameter tuning, and cost-based threshold selection.

## Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate lab05-late-deliveries
```

Install the dependencies and project package:

```bash
pip install -r requirements.txt
pip install -e .
```

## Run

Run the complete training and evaluation workflow:

```bash
python -m latedeliveries.train
```

This command compares models, tunes the selected model, chooses a threshold using training data, evaluates the reserved test set, and saves the results.

The program generates four files:

- reports/cv_results.csv - cross-validation results.
- reports/metrics.json - final test metrics and business costs.
- models/late_orders.joblib - saved machine learning pipeline.
- models/model_info.json - model settings, threshold, features, and results.

## Check it

Run the three required checks:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

All three checks passed during development, including 18 automated tests.

## Decisions

### Data and Feature Selection

The provided dataset contains 3,000 orders. Of these, 468 orders (15.60%) arrived late.

The model makes predictions when an order is placed, so only information available at that moment should be used.

Numeric features included:

- Distance in kilometres between the restaurant and customer.
- Number of items ordered.
- Hour when the order was placed.

Categorical features included:

- Weather conditions.
- Whether the order was placed on a weekday or weekend.

Excluded columns:

- Order ID: It is an identifier and not a meaningful predictive feature.
- Vehicle: The delivery vehicle may not be assigned when the order is placed.
- Courier months: The courier's experience may not be known before assignment.
- Preparation minutes: The actual preparation time is not available when the order is placed.
- Customer rating: The rating is collected after delivery and would cause data leakage.
- Late: This is the target variable, not an input feature.

The dataset was divided into 75% training data (2,250 orders) and 25% testing data (750 orders), using stratification and random seed 42.

The preprocessing pipeline uses median imputation and scaling for numerical features. Categorical features are one-hot encoded, and unseen categories are handled without errors.

### Data Leakage Experiment

I repeated the cross-validation comparison after temporarily including customer rating as a feature.

For Balanced Logistic Regression, ROC-AUC increased from 0.7680 to 0.9043, an improvement of 0.1363. However, this improvement is misleading because customer ratings are collected after delivery and are not available when an order is placed.

Therefore, customer rating was excluded from the final model. The experiment used training data only and did not change the final feature list or saved model. The code and output are included in the analysis notebook.

### Model Comparison

Five models were compared using five-fold stratified cross-validation on the training data only.

All values are reported as mean +/- standard deviation.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy baseline | 0.844 +/- 0.001 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 0.500 +/- 0.000 |
| Logistic Regression | 0.855 +/- 0.014 | 0.607 +/- 0.162 | 0.199 +/- 0.058 | 0.298 +/- 0.083 | 0.769 +/- 0.028 |
| Balanced Logistic Regression | 0.730 +/- 0.024 | 0.322 +/- 0.031 | 0.655 +/- 0.054 | 0.432 +/- 0.038 | 0.768 +/- 0.028 |
| Decision Tree | 0.752 +/- 0.019 | 0.252 +/- 0.045 | 0.299 +/- 0.061 | 0.273 +/- 0.048 | 0.567 +/- 0.032 |
| Random Forest | 0.834 +/- 0.013 | 0.447 +/- 0.069 | 0.245 +/- 0.058 | 0.313 +/- 0.061 | 0.699 +/- 0.023 |

The DummyClassifier achieved relatively high accuracy because most deliveries arrived on time, but it did not identify any late deliveries. This demonstrates why accuracy alone is not sufficient.

Balanced Logistic Regression was selected because it achieved higher recall and F1 than the other candidates, while maintaining an ROC-AUC close to standard Logistic Regression.

The company pays $4 for each flagged order and $15 for each missed late delivery. Using out-of-fold training predictions and cost-based thresholds, Balanced Logistic Regression had an estimated cost of $4,267, compared with $4,301 for standard Logistic Regression.

The $34 difference is small, so the results do not prove that Balanced Logistic Regression is consistently better. However, the business cost comparison supports its selection for further evaluation.

### Hyperparameter Tuning

GridSearchCV was used to tune Balanced Logistic Regression with five-fold stratified cross-validation.

Two hyperparameters were tested across eight combinations.

The best settings were:

- C = 0.1
- fit_intercept = True

ROC-AUC was selected as the tuning metric because it measures the model's ability to distinguish late deliveries from on-time deliveries across classification thresholds.

The untuned ROC-AUC was 0.7680, and the tuned ROC-AUC was 0.7684.

The improvement was 0.0004, which is much smaller than the original fold-to-fold standard deviation of 0.0278. Therefore, tuning did not produce a meaningful improvement.

### Classification Threshold

A classification threshold was selected using QuickBite's business costs instead of assuming that the default threshold of 0.50 was optimal.

The cost formula was:

Total cost = $4 x number of flagged orders + $15 x number of missed late deliveries.

Out-of-fold probabilities were generated using five-fold cross-validation on training data only.

Thresholds from 0.05 to 0.95, increasing by 0.05, were evaluated.

The lowest-cost threshold was 0.65.

Training-stage costs:

| Policy | Estimated cost |
|---|---:|
| Flag nobody | $5,265 |
| Flag everybody | $9,000 |
| Selected model | $4,267 |

The model reduced estimated training-stage costs by $998 compared with flagging nobody.

The test set was not used to select the model, hyperparameters, or threshold.

## Result

The final model was trained on all 2,250 training orders and evaluated on the reserved 750-order test set using threshold 0.65.

### Confusion Matrix

| Actual outcome | Predicted on time | Predicted late |
|---|---:|---:|
| On time | 568 | 65 |
| Late | 64 | 53 |

### Final Test Metrics

| Metric | Result |
|---|---:|
| Precision | 0.4492 |
| Recall | 0.4530 |
| F1-score | 0.4511 |
| ROC-AUC | 0.7636 |

### Business Costs

| Policy | Total cost |
|---|---:|
| Flag nobody | $1,755 |
| Flag everybody | $3,000 |
| Selected model | $1,432 |

The model reduced estimated costs by $323, or 18.4%, compared with flagging nobody.

This result is encouraging, but the savings are based on one simulated test dataset. They should not be treated as guaranteed savings in real delivery operations.

## What the Tests Cover

The project contains 18 passing automated tests.

The tests verify feature selection, excluded columns, reproducible stratified splitting, missing-value handling, unseen weather categories, and valid prediction probabilities.

The business cost tests use examples calculated by hand. Threshold tests check situations where flagging nobody or flagging everybody produces the lowest cost. Other tests check invalid input lengths.

The saved-model test uses the pytest tmp_path fixture to verify that predictions and probabilities remain the same after saving and reloading the complete pipeline.

Parameterized tests are also included.

Real-world changes in customer behaviour and delivery conditions were not tested because the assignment uses simulated data.

## Known Limitations

The dataset is simulated and may not represent actual QuickBite operations.

Actual preparation time and courier information were excluded because they may not be available when the order is placed. This limits the information available to the model.

The model identified 53 of the 117 late deliveries in the test set but missed 64 late deliveries.

Although the model reduced estimated business costs, further evaluation using real delivery data would be necessary before full deployment.