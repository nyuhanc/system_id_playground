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
#df_test_OG = pd.read_csv('data/faultfreetesting.txt')

# Normalize data
def normalize_data(df):
    for col in df.columns:
        if 'xmeas' in col:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        elif 'xmv' in col:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    return df

device = 'cpu' # 'cpu' or 'cuda' for gpu

# System Parameters
n_lags = 10
targets = ['xmeas_1']

# Search Parameters
par_n_trials = 1000  # number of trials

# In the  case we want to use manually increasing number of boosting rounds (see objective function)
par_min_resource = 200  # minimum number of boosting rounds (n_estimators)
par_max_resource = 1000  # maximum number of boosting rounds (n_estimators)

# Define a function that takes a value x from 0 to n_trials and returns n_estimators
# that increases from min_resource to max_resource in an exponential manner
def budget_func(x):
    return int(par_min_resource * (par_max_resource / par_min_resource)
               ** ((x + 1) / par_n_trials) ** ((5*par_n_trials)/(x+par_n_trials)))

# Generate lagged features for target
df_train = normalize_data(df_train_OG.copy())
for target in targets:
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
df = df.copy()

# Train-val-test split (80/19/1), but all dividable with 512 (chosen as max batch size)
train_size = int(0.8 * len(df))
train_size = train_size - (train_size % 512)
train_df = df[:train_size]

val_size = int(0.19 * len(df))
val_size = val_size - (val_size % 512)
val_df = df[train_size:train_size + val_size]

print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")
print(f"Number of trials: {par_n_trials}")

# Number of features
num_features = len(xmv_variables) + 1  # 11 xmv variables + 1 target variable

for target in targets:

    # Define predictors: - xmvs from time t-n_lags to t (!!!)
    #                    - xmeas from time t-(n_lags+1) to t-1
    predictors = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
    for var in xmv_variables:
        predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

    X_train = train_df[predictors].values
    y_train = train_df[target].values
    X_val = val_df[predictors].values
    y_val = val_df[target].values

    def objective(trial):

        # Use a high number of boosting rounds; optuna will automaticaly stop boosting
        # iter. if the score will fail to improve (some early stopping is used)
        n_estimators =  par_max_resource # 1000

        # Uncomment to use manually increasing number of boosting rounds
        # # Define n_estimators based on the current resource budget
        # n_estimators = budget_func(trial.number)
        # trial.set_user_attr("n_estimators", n_estimators) # save n_estimators to trial object
        # print(f"Trial {trial_num}: n_estimators = {n_estimators}")

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
            'max_depth' : trial.suggest_int("max_depth", 4, 10),
            'min_child_weight' : trial.suggest_int("min_child_weight", 2, 10),
            'eta' : trial.suggest_float("eta", 1e-8, 1.0, log=True),
            'gamma' : trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            'grow_polic' : trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"]),
            'device': device,
        }

        if device == 'cpu':
            model = xgb.XGBRegressor(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=1)  # could also use xgb.cv
            y_pred = model.predict(X_val)
        elif device == 'cuda':

            # Convert data to DMatrix format
            dtrain = xgb.DMatrix(X_train, label=y_train.reshape(-1, 1))
            dval = xgb.DMatrix(X_val, label=y_val.reshape(-1, 1))
            evals = [(dtrain, 'train'), (dval, 'eval')]

            # Note: 'fit' is renamed to 'train' in xgb.train for GPU
            # Note: 'n_estimators' is renamed to 'num_boost_round' in xgb.train
            # Note: 'early_stopping_rounds' is passed as a function argument, not a parameter
            bst = xgb.train(param, dtrain, num_boost_round=n_estimators, evals=evals,
                            early_stopping_rounds=param['early_stopping_rounds'], verbose_eval=1)
            y_pred = bst.predict(dval)
        else:
            raise ValueError(f"Unknown device: {device} (should be 'cpu' or 'cuda')")

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
        n_jobs=2 if device=='cuda' else -1, # Using more threads may cause my GPU to run out of memory (if using GPU)
        gc_after_trial=True, # Collect garbage after each trial to avoid GPU memory leak
    )

    print('Number of finished trials: {}'.format(len(study.trials)))
    print('Best trial:')
    trial = study.best_trial

    print('  Best MSE: {}'.format(trial.value))
    print(f"Best hyperparameters for {target}:")

    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # Save results to csv, add a unique 3-digit timestamp to the filename
    timestr = time.strftime("%m%d-%H%M")
    results = study.trials_dataframe()
    results.to_csv(f'results/xgb_OptunaHyperband_({timestr})_{target}.csv', index=False)

    # Following https://www.kaggle.com/code/hamzaghanmi/xgboost-catboost-using-optuna#3.-XGBoost-using-Optuna-
    # Plot the optimization history
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_image(f"results/xgb_OptunaHyperband_({timestr})_{target}_optimization_history.png")

    # Plot the importance of the parameters
    fig = optuna.visualization.plot_param_importances(study)
    fig.write_image(f"results/xgb_OptunaHyperband_({timestr})_{target}_param_importances.png")

    # Plot the slice plot
    fig = optuna.visualization.plot_slice(study)
    fig.write_image(f"results/xgb_OptunaHyperband_({timestr})_{target}_slice_plot.png")

    # Plot the parallel plot
    fig = optuna.visualization.plot_parallel_coordinate(study)
    fig.write_image(f"results/xgb_OptunaHyperband_({timestr})_{target}_parallel_plot.png")

    # Plot the contour plot
    fig = optuna.visualization.plot_contour(study)
    fig.write_image(f"results/xgb_OptunaHyperband_({timestr})_{target}_contour_plot.png")

    # Plot the edf plot
    fig = optuna.visualization.plot_edf(study)
    fig.write_image(f"results/xgb_OptunaHyperband_({timestr})_{target}_edf_plot.png")

