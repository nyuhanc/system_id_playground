# Hyperparameter tuning using scikit-learn GridSearchCV (varying two hypers at others fixed at optimal values
# found by HalvingRandomSearchCV)
# This is a secondary analysis, i.e. we are not looking for the best hyperparameters, but rather
# we are looking for the dependence of the loss on two hyperparameters while keeping the others
# fixed at their optimal values found by HalvingRandomSearchCV.

import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.model_selection import GridSearchCV

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
df = df.copy()

# Train-val-test split (80/19/1), but all dividable with 512 (consistency with LSTM)
# We use train+val set for CV
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
    'n_estimators': [100, 150, 200, 250, 300], # [100, 150, 200, 250, 300]
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8], # [1, 2, 3, 4, 5, 6, 7, 8]
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.2], # [0.01, 0.05, 0.1, 0.15, 0.2]
    'sub': [0.5, 0.75, 1], # [0.5, 0.75, 1] subsample
    'colsample_bytree': [0.5, 0.7, 0.9], # [0.5, 0.7, 0.9]
}

# Chosen (or promising) set of parameters (from HRSCV)
param_grid_best = {
    'sub': [1],
    'n_estimators': [100],
    'max_depth': [8],
    'learning_rate': [0.2],
    'colsample_bytree': [0.9]
}

# For each of two hyperparameters, perform a grid search while keeping the other hyperparameters fixed
# Then, plot the loss as a function of the hyperparameter being varied as a heatmap
combinations = []
for key1 in param_grid_best.keys():
    for key2 in param_grid_best.keys():
        if key1 == key2:
            continue # skip if the two hyperparameters are the same
        # skip if the combination already tested
        elif (key1, key2) in combinations or (key2, key1) in combinations:
            continue
        else:
            # Add the combination to the list of combinations
            combinations.append((key1, key2))

            # Define parameter grid
            # Only the two hyperparameters being varied are included in the grid effectively
            param_grid_ = {
                key1: param_grid[key1],
                key2: param_grid[key2],
            }
            # Other hyperparameters are fixed and taken from param_grid defined above
            for key in param_grid_best.keys():
                if key not in [key1, key2]:
                    param_grid_[key] = param_grid_best[key]

            # Instantiate the grid search model
            grid_search = GridSearchCV(
                estimator=xgb.XGBRegressor(),
                param_grid=param_grid_,
                scoring='neg_mean_squared_error',
                n_jobs=-2,
                cv=4,
                verbose=2,
            )

            # Fit the grid search to the data
            grid_search.fit(X_train_val, y_train_val)

            # Print name of the hyperparameters being varied
            print(f'Hyperparameters being varied: {key1}, {key2}')

            # Print best parameters
            print('Best parameters: ')
            print(grid_search.best_params_)

            # Print best score
            print('Best score: ')
            print(grid_search.best_score_)

            # Save results to csv, add a unique timestamp to the filename
            timestr = time.strftime("%m%d-%H%M")
            results = pd.DataFrame(grid_search.cv_results_)
            results.to_csv(f'results/xgb_SGCV_2by2_({timestr})_{target}_{key1}_{key2}.csv', index=False)

            # Plot loss as a function of the hyperparameter being varied as a heatmap
            # Extract the loss values
            loss = results['mean_test_score'].values
            # Extract the hyperparameter values
            hyper1 = results['param_' + key1].values
            hyper2 = results['param_' + key2].values
            # Reshape the loss and hyperparameter values into a 2D array, make sure the
            # hyperparameters are in the correct order by examining hyper1 and hyper2
            if hyper1[0] == hyper1[1]:
                loss = loss.reshape(len(param_grid[key2]), len(param_grid[key1]))
                hyper1 = hyper1.reshape(len(param_grid[key2]), len(param_grid[key1]))
                hyper2 = hyper2.reshape(len(param_grid[key2]), len(param_grid[key1]))
            else:
                loss = loss.reshape(len(param_grid[key1]), len(param_grid[key2]))
                hyper1 = hyper1.reshape(len(param_grid[key1]), len(param_grid[key2]))
                hyper2 = hyper2.reshape(len(param_grid[key1]), len(param_grid[key2]))

            # Plot the heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(loss, annot=True, fmt='.4f', xticklabels=param_grid[key2], yticklabels=param_grid[key1])
            plt.xlabel(key2)
            plt.ylabel(key1)
            plt.savefig(f'plots/xgb_SGCV_2by2_({timestr})_{target}_{key1}_{key2}.pdf', format='pdf', dpi=1200)
            plt.show()














