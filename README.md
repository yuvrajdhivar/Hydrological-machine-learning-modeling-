# Generic Hydrological Regression Scripts

This repository contains two reusable Python workflows for regression modelling:

- `svr_regression.py` — Support Vector Regression (RBF kernel)
- `xgboost_regression.py` — XGBoost regression

The scripts are generalized from the supplied study scripts. They do not contain basin-, station-, or project-specific input/output filenames.

## 1. Requirements

Python 3.x and the packages listed in `requirements.txt`.

Install packages:

```bash
pip install -r requirements.txt
```

## 2. Prepare the input data

Place a CSV file in this folder. The CSV should contain:
- one target/observed variable column (for example, `target` or `Qobs`);
- one or more numeric predictor columns;
- optionally, a date column for chronological sorting.

Example structure:

| Date | predictor_1 | predictor_2 | target |
|---|---:|---:|---:|
| 2001-01-01 | 12.4 | 3.2 | 45.1 |
| 2001-01-02 | 10.8 | 2.9 | 41.7 |

Do not upload confidential, restricted, or unpublished research data to a public repository.

## 3. Configure each script

At the top of each Python script, edit:

```python
DATA_FILE = "your_dataset.csv"
TARGET_COLUMN = "target"
DATE_COLUMN = "Date"  # Use None if there is no date column
TRAIN_FRACTION = 0.70
```

The scripts use a chronological split: the first 70% of records for training and the remaining 30% for testing. Change `TRAIN_FRACTION` if a different split is required.

The SVR script standardizes predictor variables using a scaler fitted only on the training data. The XGBoost script does not standardize predictors because tree-based models do not require feature scaling.

## 4. Run the models

```bash
python svr_regression.py
python xgboost_regression.py
```

## 5. Outputs

The scripts create separate output folders:

- `outputs_svr/`
- `outputs_xgboost/`

Each folder contains the trained model, predictions for the full chronological period, and train/test performance metrics in JSON format. The SVR folder also contains the fitted predictor scaler.

## 6. Notes

- The target column and predictor columns must be numeric, except for the optional date column.
- Rows with missing or infinite values in the selected predictors or target are excluded.
- Model hyperparameters can be changed in the configuration dictionaries near the top of each script.
- Performance metrics include RMSE, MAE, R², NSE, KGE, RSR, and PBIAS.
- Check the assumptions and suitability of the metrics for the variables and application in your study before interpreting the results.
