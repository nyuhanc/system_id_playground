import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error

import inspect

# Load data
df_train_OG = pd.read_csv('data/faultfreetraining.txt')
df_test_OG = pd.read_csv('data/faultfreetesting.txt')

# Normalize data
def normalize_data(df):
    for col in df.columns:
        if 'xmeas' in col:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        elif 'xmv' in col:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

# Parameters
n_lags = 3
target = 'xmeas_2'

# Generate lagged features for target
df_train = normalize_data(df_train_OG.copy())
for lag in range(1, n_lags + 1):
    df_train[f"{target}_lag{lag}"] = df_train[target].shift(lag)

# Generate lagged features for all xmv_j
xmv_variables = [col for col in df_train.columns if 'xmv' in col]

for var in xmv_variables:
    for lag in range(1, n_lags + 1):
        df_train[f"{var}_lag{lag}"] = df_train[var].shift(lag)

# Drop missing values (due to lagged features creation)
df = df_train.dropna()

# Train-test split (80/20)
train_size = int(0.8 * len(df))
train_df, test_df = df[:train_size], df[train_size:]

# Define predictors
predictors = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
for var in xmv_variables:
    predictors.extend([var] + [f"{var}_lag{i}" for i in range(1, n_lags + 1)])

X_train = train_df[predictors]
y_train = train_df[target]
X_test = test_df[predictors]
y_test = test_df[target]

# Create XGBoost model
model = xgb.XGBRegressor(
    n_estimators=150,
    learning_rate=0.2,
    objective='reg:squarederror',
    max_depth=5,
    max_leaves=None,
    max_bin=None,
    grow_policy=None,
    verbosity=None,
    booster=None,
    tree_method=None,
    gamma=None,
    min_child_weight=None,
    max_delta_step=None,
    subsample=None,
    sampling_method=None,
    colsample_bytree=None,
    colsample_bylevel=None,
    colsample_bynode=None,
    reg_alpha=None,
    reg_lambda=None,
    scale_pos_weight=None,
    base_score=None,
    num_parallel_tree=None,
    random_state=None,
    n_jobs=None,
    monotone_constraints=None,
    interaction_constraints=None,
    importance_type=None,
    gpu_id=None,
    validate_parameters=None,
    predictor=None,
    enable_categorical=False,
    feature_types=None,
    max_cat_to_onehot=None,
    max_cat_threshold=None,
    eval_metric=None,
    early_stopping_rounds=None,
    callbacks=None,
)


# Train XGBoost model
model.fit(X_train, y_train)

# Recursive forecasting
predictions = []
xmeas_lags = y_test.iloc[:n_lags].tolist()

predict_n_steps = 100

for i in range(predict_n_steps): # len(y_test) - n_lags):
    # Prepare input data for prediction
    input_data = {f"{target}_lag{j + 1}": xmeas_lags[-(j + 1)] for j in range(n_lags)}
    for var in xmv_variables:
        input_data[var] = test_df.iloc[i + n_lags][var]
        for lag in range(1, n_lags + 1):
            input_data[f"{var}_lag{lag}"] = test_df.iloc[i + n_lags - lag][var]
    input_df = pd.DataFrame([input_data])

    # Make a prediction
    prediction = model.predict(input_df)[0]
    predictions.append(prediction)

    # Update xmeas_1 lags with the new prediction
    xmeas_lags.append(prediction)

# Calculate RMSE
rmse_value = mean_squared_error(y_test.iloc[n_lags:n_lags+predict_n_steps], predictions, squared=False)

# Plot predictions vs. actuals
plot_samples = predict_n_steps
actual_values = y_test[n_lags:n_lags+plot_samples]
predicted_values = predictions[:plot_samples]

# Create a figure and axis object
fig, ax = plt.subplots()

# Plot actual and predicted values
ax.plot(np.arange(len(predicted_values)), actual_values, label='Actual')
ax.plot(np.arange(len(predicted_values)), predicted_values, label='Predicted')

# Add grid, legend, and RMSE text
ax.grid(True)
ax.legend()
ax.text(0.985, 0.02, f'RMSE: {rmse_value:.4f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# Show the plot
plt.show()