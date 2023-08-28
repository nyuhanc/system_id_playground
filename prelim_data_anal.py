import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error

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

# Generate lagged features for target
df_train = normalize_data(df_train_OG.copy())

# Compute cross-correlation between variables in the training set, i.e., between all xmeas and all xmvs
# This is to see if there are any strong correlations between xmeas and xmvs

# for i in df_train[[f'xmeas_{j}' for j in range(1, 41+1)]+[f'xmv_{j}' for j in range(1, 11+1)]]:
#     for j in df_train[[f'xmeas_{j}' for j in range(1, 41+1)]+[f'xmv_{j}' for j in range(1, 11+1)]]:
#         print(f'Correlation between {i} and {j}: {df_train[i].corr(df_train[j])}')

# represent cross-correlation matrix as a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df_train[[f'xmeas_{j}' for j in range(1, 41+1)]+[f'xmv_{j}' for j in range(1, 11+1)]].corr())
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.savefig('plots/cross_correlation_matrix.pdf', format='pdf', dpi=1200)
plt.show()




