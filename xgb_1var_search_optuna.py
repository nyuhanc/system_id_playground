import time
import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load data
df_train_OG = pd.read_csv('data/faultfreetraining.txt')

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
target = 'xmeas_1'

# Generate lagged features for target
df_train = normalize_data(df_train_OG.copy())
for lag in range(1, n_lags + 1):
    df_train[f"{target}_lag{lag}"] = df_train[target].shift(lag)

# Generate lagged features for all xmv_j
xmv_variables = [col for col in df_train.columns if 'xmv' in col]  # xmvs 1-11

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

# Split data for validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def objective(trial):
    # Define hyperparameter search space using trial object
    param = {
        'objective': 'reg:squarederror',
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 1, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2),
        'subsample': trial.suggest_float('subsample', 0.5, 1),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'early_stopping_rounds': 50  # This parameter is set to stop training when validation score stops improving
    }
    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=3)
    y_pred = model.predict(X_val)

    return mean_squared_error(y_val, y_pred)

# Create a study object and specify the direction is 'minimize' for MSE
study = optuna.create_study(
    direction='minimize', # minimize MSE
    pruner=optuna.pruners.HyperbandPruner(
    min_resource=100, # Minimum number of boosting rounds
    max_resource=500, # Maximum number of boosting rounds
    reduction_factor=3 # Cut the number of epochs by 3 after each iteration


    ) # Use Hyperband pruning
)
study.optimize(
    objective,
    n_trials=100,
    # n_jobs=1, # allways uses all cores since xgboost uses all cores
)

# Print best parameters
print('Best parameters: ')
print(study.best_params)
print('\n')

# Print best score
print('Best MSE: ', study.best_value)

# Save results to csv, add a unique 3-digit timestamp to the filename
timestr = time.strftime("%m%d-%H%M")
results = study.trials_dataframe()
results.to_csv(f'results/xgb_OptunaHyperband_({timestr})_{target}.csv', index=False)
