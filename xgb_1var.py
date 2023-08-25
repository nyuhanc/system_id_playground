import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load data
df_train_OG = pd.read_csv('data/faultfreetraining.txt')
df_test_OG = pd.read_csv('data/faultfreetesting.txt')

# Parameters
n_lags = 3
target = 'xmeas_1'

# Generate lagged features for target
df_train = df_train_OG.copy()
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
    for lag in range(1, n_lags + 1):
        predictors.append(f"{var}_lag{lag}")

X_train = train_df[predictors]
y_train = train_df[target]
X_test = test_df[predictors]
y_test = test_df[target]

# Train XGBoost model
model = xgb.XGBRegressor()
model.fit(X_train, y_train)

# Recursive forecasting
predictions = []
xmeas_lags = y_test.iloc[:n_lags].tolist()

predict_n_steps = 100

for i in range(predict_n_steps):#len(y_test) - n_lags):
    # Prepare input data for prediction
    input_data = {f"{target}_lag{j + 1}": xmeas_lags[-(j + 1)] for j in range(n_lags)}
    for var in xmv_variables:
        for lag in range(1, n_lags + 1):
            input_data[f"{var}_lag{lag}"] = test_df.iloc[i + n_lags - lag][var]
    input_df = pd.DataFrame([input_data])

    # Make a prediction
    prediction = model.predict(input_df)[0]
    predictions.append(prediction)

    # Update xmeas_1 lags with the new prediction
    xmeas_lags.append(prediction)

# Calculate RMSE
rmse = mean_squared_error(y_test.iloc[n_lags:n_lags+predict_n_steps], predictions, squared=False)

print(rmse)


# Plot predictions vs. actuals
plot_samples = predict_n_steps
plt.plot(np.arange(len(predictions))[:plot_samples], y_test[n_lags:n_lags+plot_samples], label='Actual')
plt.plot(np.arange(len(predictions))[:plot_samples], predictions[:plot_samples], label='Predicted')
plt.grid(True)
plt.legend()
plt.show()