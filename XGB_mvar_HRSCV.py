# Hyperparameter tuning using scikit-learn HalvingRandomSearchCV for multiple targets

import time
import pandas as pd
import xgboost as xgb
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV

# Load data
df_train_OG = pd.read_csv('data/faultfreetraining.txt')


# Same normalization function as before
def normalize_data(df):
    for col in df.columns:
        if 'xmeas' in col:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        elif 'xmv' in col:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df


# Parameters
n_lags = 10
# Your list of targets
# Including targets with relatively good 1var fit
targets = [
    'xmeas_1',   # Very good 1var fit
    'xmeas_7',  # Not that good 1var fit
    'xmeas_10',  # Very good 1var fit
    'xmeas_12',  # Very good 1var fit
    'xmeas_13',  # Not that good 1var fit
    'xmeas_15',  # Very good 1var fit
    'xmeas_16',  # Not that good 1var fit
    'xmeas_17',  # Very good 1var fit
    'xmeas_18',  # Not that good 1var fit
    'xmeas_19',  # Not that good 1var fit
    'xmeas_20',  # Not that good 1var fit
    'xmeas_21',  # Not that good 1var fit
]

# Generate lagged features for target
df_train = normalize_data(df_train_OG.copy())
for target in targets:
    for lag in range(1, n_lags + 1):
        df_train[f"{target}_lag{lag}"] = df_train[target].shift(lag)

# Generate lagged features for all xmv_j
xmv_variables = [col for col in df_train.columns if 'xmv' in col] # xmvs 1-11
for var in xmv_variables:
    for lag in range(0, n_lags):
        df_train[f"{var}_lag{lag}"] = df_train[var].shift(lag)

# Drop missing values (due to lagged features creation)
df = df_train.dropna()

# Defragment the dataframe
df = df.copy()

# Train-val-test split (80/19/1), but all dividable with 512 (consistency with LSTM)
# We use train+val set for CV
train_size = int(0.8 * len(df))
train_size = train_size - (train_size % 512)
val_size = int(0.19 * len(df))
val_size = val_size - (val_size % 512)
train_val_df = df[:train_size + val_size]

# Define predictors
predictors = []
for target in targets:
    predictors.extend([f"{target}_lag{i}" for i in range(1, n_lags + 1)])
for var in xmv_variables:
    predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

X_train_val = train_val_df[predictors].values
y_train_val = train_val_df[targets].values  # Multiple targets here

# Define parameter grid
param_grid = {
    # 'n_estimators' is the boosting rounds, which is the resource metric (*)
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2],
    #'subsample': [0.5, 0.75, 1],
    'colsample_bytree': [0.9, 0.95, 1.0],
    'reg_lambda': [1e-3, 1e-2, 1e-1],  # L2
    'reg_alpha': [1e-3, 1e-2, 1e-1],  # L1
    #'booster': ['gbtree', 'gblinear', 'dart'],
    'min_child_weight': [1, 3, 5, 7, 9],
    'gamma': [0.0, 0.1, 0.2],  # Minimum loss reduction required to make a further partition on a leaf node
}

min_resources = 100  # Minimum number of boosting rounds
max_resources = 500  # Maximum number of boosting rounds
# Number of resources is multiplied by factor after each iteration. The exponent of
# 1/n means that the algorithm will increase the number of boosting rounds n+1 times
factor = (max_resources/min_resources) ** (1/4)  # 5 iterations for (1/4) exponent

# Instantiate the halving random search model
search = HalvingRandomSearchCV(
    estimator=xgb.XGBRegressor(
        n_jobs=1 # Use only one thread for each model
    ),
    param_distributions=param_grid,
    resource='n_estimators',  # Use boosting rounds as resource metric (*)
    min_resources=min_resources,
    max_resources=max_resources+1,
    factor=factor,
    n_candidates=5,  # Number of candidates in the first iteration (then ~ this/factor in each iteration)
    scoring='neg_mean_squared_error',
    n_jobs=10, # More threads can lead to memory overflow (for the multivar case)
    cv=4,  # 4-fold split CV
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
results.to_csv(f'results/xgb_mvar_HRSCVres_({timestr})_{target}.csv', index=False)

