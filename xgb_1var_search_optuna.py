# Objective: Optimize XGBoost hyperparameters for 1-variable prediction using Optuna
# Optuna is a black-box optimizer, which means it needs an objectivefunction, which
# returns a numerical value to evaluate the performance of the hyperparameters, and
# decide where to sample in upcoming trials. See
# https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_cv.py
# Also see: https://optuna.readthedocs.io/en/stable/index.html and
# https://xgboost.readthedocs.io/en/stable/parameter.html

# Problem: Don't know how to connect min/max_resource to n_estimators through Optuna,
# thus it is done manually in the objective function through trial.number

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

# System Parameters
n_lags = 10
target = 'xmeas_1'

# Search Parameters
par_n_trials = 1000
par_min_resource = 100  # minimum number of boosting rounds (n_estimators)
par_max_resource = 500  # maximum number of boosting rounds (n_estimators)

# Define a function that takes a value x from 0 to n_trials and returns n_estimators
# that increases from min_resource to max_resource in an exponential manner
def budget_func(x):
    return int(par_min_resource * (par_max_resource / par_min_resource)
               ** ((x + 1) / par_n_trials) ** ((5*par_n_trials)/(x+par_n_trials)))


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
    # Get the current resource budget for this trial
    trial_num = trial.number

    # Define n_estimators based on the current resource budget
    n_estimators = budget_func(trial_num)
    trial.set_user_attr("n_estimators", n_estimators) # save n_estimators to trial object
    print(f"Trial {trial_num}: n_estimators = {n_estimators}")

    # Define hyperparameter search space using trial object
    param = {
        'objective': 'reg:squarederror',
        "booster": trial.suggest_categorical("booster", ["gbtree"]),  # , "gblinear", "dart"]),
        'n_estimators': n_estimators, # number of boosting rounds
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3), # step size shrinkage used in update to prevents overfitting
        'subsample': trial.suggest_float('subsample', 0.5, 1), # subsample ratio of the training instances
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1), # subsample ratio of columns when constructing each tree
        'early_stopping_rounds': 50,  # This parameter is set to stop training when validation score stops improving
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        #########################
        'device': 'cuda',
    }

    if param["booster"] == "gbtree" or param["booster"] == "dart":
        param["max_depth"] = trial.suggest_int("max_depth", 4, 10)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])

    if param["booster"] == "dart":
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)


    model = xgb.XGBRegressor(**param)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=0)  # could also use xgb.cv
    y_pred = model.predict(X_val)

    return mean_squared_error(y_val, y_pred)

# Create a study object and specify the direction is 'minimize' for MSE
study = optuna.create_study(
    direction='minimize',  # minimize MSE
    pruner=optuna.pruners.HyperbandPruner(),  # Use Hyperband Pruner
    sampler=optuna.samplers.TPESampler()  # Use TPE sampler (should be by default)
)
study.optimize(
    objective,
    n_trials=par_n_trials,
    n_jobs=1, # Using more threads may cause my GPU to run out of memory
    gc_after_trial=True, # Collect garbage after each trial to avoid GPU memory leak
)

print('Number of finished trials: {}'.format(len(study.trials)))
print('Best trial:')
trial = study.best_trial

print('  Best MSE: {}'.format(trial.value))
print('  Params: ')

for key, value in trial.params.items():
    print('    {}: {}'.format(key, value))

# Save results to csv, add a unique 3-digit timestamp to the filename
timestr = time.strftime("%m%d-%H%M")
results = study.trials_dataframe()
results.to_csv(f'results/xgb_OptunaHyperband_({timestr})_{target}.csv', index=False)
