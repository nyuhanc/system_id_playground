# Hyperparameter tuning using scikit-learn HalvingRandomSearchCV

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import HalvingRandomSearchCV  # <-- Import HalvingRandomSearchCV

# Load data
df_train_OG = pd.read_csv('data/faultfreetraining.txt')
#df_test_OG = pd.read_csv('data/faultfreetesting.txt')

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
xmv_variables = [col for col in df_train.columns if 'xmv' in col] # xmvs 1-11

for var in xmv_variables:
    for lag in range(0, n_lags + 1):
        df_train[f"{var}_lag{lag}"] = df_train[var].shift(lag)

# Drop missing values (due to lagged features creation)
df = df_train.dropna()

# Defragment the dataframe
df_train = df_train.copy()

# Define predictors: xmvs from time t-n_lags to t (!!!), and xmeas from time t-n_lags to t-1
predictors = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
for var in xmv_variables:
    predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags + 1)])

X_train = df_train[predictors]
y_train = df_train[target]

# Use scikit-learn RandomizedSearchCV to tune hyperparameters

# Create XGBoost model
model = xgb.XGBRegressor()

# Define parameter grid
param_grid = {
    'n_estimators': [100, 150, 200, 250, 300], # [100, 150, 200, 250, 300]
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8], # [1, 2, 3, 4, 5, 6, 7, 8]
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2], # [0.01, 0.05, 0.1, 0.15, 0.2]
    'sub': [0.5, 0.75, 1], # [0.5, 0.75, 1] subsample
    'colsample_bytree': [0.5, 0.7, 0.9], # [0.5, 0.7, 0.9]
}

# Instantiate the halving random search model
halving_random_search = HalvingRandomSearchCV(  # <-- Use HalvingRandomSearchCV
    estimator=model,
    param_distributions=param_grid,
    n_candidates="exhaust",  # This will exhaustively try all candidates in param_grid
    factor=2,  # Reduction factor for the number of candidates/iterations
    scoring='neg_mean_squared_error',
    n_jobs=-2,
    cv=5,
    verbose=3,
)

# Fit the halving random search to the data
halving_random_search.fit(X_train, y_train)

# Print best parameters
print('Best parameters: ')
print(halving_random_search.best_params_)
print('\n')

# Print best estimator
print('Best estimator: ', halving_random_search.best_estimator_)

# Print best score
print('with score: ', halving_random_search.best_score_)

# Save results to csv, add a unique 3-digit timestamp to the filename
timestr = time.strftime("%m%d-%H%M")
results = pd.DataFrame(halving_random_search.cv_results_)
results.to_csv(f'results/xgb_HRSCVres_({timestr})_{target}.csv', index=False)





