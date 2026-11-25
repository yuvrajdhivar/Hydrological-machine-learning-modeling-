# ===============================
# 1. Import Libraries
# ===============================
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVR
import joblib

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR


# ===============================
# 2. Load Data
# ===============================
df = pd.read_csv('input.csv', parse_dates=['Date'], index_col='Date')

feature_cols = [col for col in df.columns if col != 'Qobs']
X = df[feature_cols].values
y = df['Qobs'].values


# ===============================
# 3. Chronological Train-Test Split (75/25)
# ===============================
split = int(0.70 * len(X))

X_train_raw = X[:split]
X_test_raw  = X[split:]

y_train_raw = y[:split]
y_test_raw  = y[split:]


# ===============================
# 4. Scale (Optional for XGBoost but kept for consistency)
# ===============================
scaler_X = StandardScaler()

X_train = scaler_X.fit_transform(X_train_raw)
X_test  = scaler_X.transform(X_test_raw)

y_train = y_train_raw
y_test  = y_test_raw


# ===============================
# 5. Build SVR Model
# ===============================

model = SVR(

    kernel='rbf',

    C=1500,

    epsilon=0.008,

    gamma=0.006
)

model.fit(X_train, y_train)
# ===============================
# 6. Predict
# ===============================
y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)

y_train_true = y_train
y_test_true  = y_test


# ===============================
# 7. Hydrological Metrics
# ===============================
def NSE(obs, sim):
    return 1 - np.sum((obs - sim)**2) / np.sum((obs - np.mean(obs))**2)

def PBIAS(obs, sim):
    return 100 * np.sum(obs - sim) / np.sum(obs)

# RMSE
train_rmse = np.sqrt(mean_squared_error(y_train_true, y_train_pred))
test_rmse  = np.sqrt(mean_squared_error(y_test_true, y_test_pred))

# R²
train_r2 = r2_score(y_train_true, y_train_pred)
test_r2  = r2_score(y_test_true, y_test_pred)

# NSE
train_nse = NSE(y_train_true, y_train_pred)
test_nse  = NSE(y_test_true, y_test_pred)

# PBIAS
train_pbias = PBIAS(y_train_true, y_train_pred)
test_pbias  = PBIAS(y_test_true, y_test_pred)


print("\n===== SVR MODEL PERFORMANCE =====")
print(f"Train RMSE  : {train_rmse:.2f} | Test RMSE  : {test_rmse:.2f}")
print(f"Train R²    : {train_r2:.3f} | Test R²    : {test_r2:.3f}")
print(f"Train NSE   : {train_nse:.3f} | Test NSE   : {test_nse:.3f}")
print(f"Train PBIAS : {train_pbias:.2f}% | Test PBIAS : {test_pbias:.2f}%")
# MAE already available from sklearn if needed

def RSR(obs, sim):
    rmse = np.sqrt(np.mean((obs - sim) ** 2))
    std_obs = np.std(obs)
    return rmse / std_obs

def KGE(obs, sim):
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

# MAE
train_mae = mean_absolute_error(y_train_true, y_train_pred)
test_mae  = mean_absolute_error(y_test_true, y_test_pred)

# RSR
train_rsr = RSR(y_train_true, y_train_pred)
test_rsr  = RSR(y_test_true, y_test_pred)

# KGE
train_kge = KGE(y_train_true, y_train_pred)
test_kge  = KGE(y_test_true, y_test_pred)


print("\n===== SVR MODEL PERFORMANCE =====")

print(f"Train RMSE  : {train_rmse:.2f} | Test RMSE  : {test_rmse:.2f}")
print(f"Train MAE   : {train_mae:.2f} | Test MAE   : {test_mae:.2f}")

print(f"Train R²    : {train_r2:.3f} | Test R²    : {test_r2:.3f}")
print(f"Train NSE   : {train_nse:.3f} | Test NSE   : {test_nse:.3f}")

print(f"Train KGE   : {train_kge:.3f} | Test KGE   : {test_kge:.3f}")
print(f"Train RSR   : {train_rsr:.3f} | Test RSR   : {test_rsr:.3f}")

print(f"Train PBIAS : {train_pbias:.2f}% | Test PBIAS : {test_pbias:.2f}%")

# ===============================
# 8. Save Model & Predictions
# ===============================
joblib.dump(model, 'output_svr_model.pkl')

#pd.DataFrame({
 #   'True': y_test_true,
 #   'Pred': y_test_pred
#}).to_csv('xgboost_predictions.csv', index=False)

 #("\n✓ Saved: xgboost_inflow_model.json")
# ("✓ Saved: xgboost_predictions.csv")

# 10. Combine FULL PERIOD (Train + Test)
# ===============================

# Combine predictions
y_pred_full = np.concatenate([y_train_pred, y_test_pred])
y_true_full = np.concatenate([y_train_true, y_test_true])

# Get corresponding dates
dates_full = df.index[:len(y_pred_full)]

# Create dataframe
full_results = pd.DataFrame({
    'Date': dates_full,
    'Observed': y_true_full,
    'Predicted': y_pred_full
})

# Save as Excel
full_results.to_excel('output_svr_results.xlsx', index=False)