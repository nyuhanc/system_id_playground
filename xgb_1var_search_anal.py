# Interpret the results of the grid search cross validation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


# --- ANALYSIS OF THE RESULTS OF THE GRID SEARCH CROSS VALIDATION -------
# --- (PRIMARY ANALYSIS) ------------------------------------------------

# Load SearchGridCV results (RandomizedSearchCV)
table_name = 'xgb_SGCV_(0829-1301)_xmeas_2.csv'
df = pd.read_csv(f'results/{table_name}')

# Extract the best parameters
best_params = df.loc[df['rank_test_score'] == 1, 'params'].values[0]
print(f'Best parameters: {best_params}')

# Extract the best score
best_score = df.loc[df['rank_test_score'] == 1, 'mean_test_score'].values[0]
print(f'Best score: {best_score}')

# Correlate mean_test_score with all hyperparameters
hypers = [
    'param_n_estimators',
    'param_max_depth',
    'param_learning_rate',
    'param_sub',
    'param_colsample_bytree',
]
for col in df.columns:
    if col in hypers:
        # Correlate col with mean_test_score
        print(f'Correlation between {col} and mean_test_score: {df[col].corr(df["mean_test_score"])}')


# Print the best 10 sets of hyperparameters
# Set print options to display full content of the 'params' column
pd.set_option('display.max_colwidth', None)
# rank by decreasing mean_test_score
df.sort_values(by='mean_test_score', ascending=False, inplace=True)
# print the first five rows
print(df[['params', 'mean_test_score']].head(10))

# --- ANALYSIS OF THE RESULTS OF THE GRID SEARCH CROSS VALIDATION -------
# --- (DEPENDENCE OF LOSS ON TWO HYPERS. WHILE OTHERS BEING FIXED) ------

# For the time being in xgb_1var_SGCV_2by2.py








