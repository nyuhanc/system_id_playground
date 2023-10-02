# Hyperparameter tuning using scikit-learn RandomizedSearchCV

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV, RandomizedSearchCV


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
n_lags = 10
target = 'xmeas_1'

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

# Train-val-test split (80/19/1), but all dividable with 512 (consistency with LSTM)
train_size = int(0.8 * len(df))
train_size = train_size - (train_size % 512)
val_size = int(0.19 * len(df))
val_size = val_size - (val_size % 512)
train_val_df = df[:train_size + val_size]

# Number of features
num_features = len(xmv_variables) + 1  # 11 xmv variables + 1 target variable

# Define predictors: - xmvs from time t-n_lags to t (!!!)
#                    - xmeas from time t-(n_lags+1) to t-1
predictors = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
for var in xmv_variables:
    predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

X_train_val = train_val_df[predictors].values
y_train_val = train_val_df[target].values

# Define parameter grid
param_grid = {
    #'n_estimators': [100, 150, 200, 250, 300],  # [100, 150, 200, 250, 300]
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],  # [1, 2, 3, 4, 5, 6, 7, 8]
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],  # [0.01, 0.05, 0.1, 0.15, 0.2]
    #'sub': [0.5, 0.75, 1],  # [0.5, 0.75, 1] subsample
    'colsample_bytree': [0.5, 0.7, 0.9],  # [0.5, 0.7, 0.9]
}

# # Instantiate the random search model
# search = RandomizedSearchCV(
#     estimator=xgb.XGBRegressor(),
#     param_distributions=param_grid,
#     n_iter=1000,  # Number of candidates/iterations
#     scoring='neg_mean_squared_error',
#     n_jobs=-2,
#     cv=5,  # 5 fold split
#     verbose=3,
# )

# Instantiate the halving random search model
search = HalvingRandomSearchCV(
    estimator=xgb.XGBRegressor(),
    param_distributions=param_grid,
    resource='n_estimators',  # Use boosting rounds as resource metric
    min_resources=100,  # Minimum number of boosting rounds
    max_resources=500,  # Maximum number of boosting rounds
    factor=2,  # Halving factor
    n_candidates=100,  # Number of candidates in the first iteration
    scoring='neg_mean_squared_error',
    n_jobs=-2,  # Use all but one CPU core
    cv=5,  # 5 fold split CV
    verbose=2,
)

# Fit the halving random search to the data
search.fit(X_train_val, y_train_val)

# Print best parameters
print('Best parameters: ')
print(search.best_params_)
print('\n')

# Print best estimator
print('Best estimator: ', search.best_estimator_)

# Print best score
print('with score: ', search.best_score_)

# Save results to csv, add a unique 3-digit timestamp to the filename
timestr = time.strftime("%m%d-%H%M")
results = pd.DataFrame(search.cv_results_)
results.to_csv(f'results/xgb_HRSCVres_({timestr})_{target}.csv', index=False)





