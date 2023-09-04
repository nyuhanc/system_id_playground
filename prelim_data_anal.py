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

# # Compute cross-correlation between variables in the training set, i.e., between all xmeas and all xmvs
# plt.figure(figsize=(12, 10))
# sns.heatmap(df_train[[f'xmeas_{j}' for j in range(1, 41+1)]+[f'xmv_{j}' for j in range(1, 11+1)]].corr())
# plt.xticks(fontsize=9)
# plt.yticks(fontsize=9)
# plt.savefig('plots/cross_correlation_matrix.pdf', format='pdf', dpi=1200)
# plt.show()

# Compute cross-correltaion between xmv variables and lagged xmeas variables
# For each xmv variable,compute cross correlation with lagged xmeas variables, each for at max 10 lags
# For each xmv variable, find the lag with the highest cross correlation

# Parameters
max_n_lags = 100 # max number of lags to consider
targets = ['xmeas_2'] #[f'xmeas_{i}' for i in range(1, 41+1)]
xmv_variables = [f'xmv_{i}' for i in range(1, 11+1)]




# For each xmeas, compute cross correlation with each xmv for each lag
for target in targets:

    # Plot cross correlation
    plt.figure(figsize=(14, 10))
    plt.title(f"Cross correlation between {target} and xmvs")
    plt.xlabel("Lag in xmv")
    plt.ylabel("Cross correlation")
    plt.grid(True)


    hci = []
    for xmv in xmv_variables:
        cors = []
        for lag in range(0, max_n_lags + 1):
            cor = np.abs(np.asarray(df_train[target]) @ np.roll(np.asarray(df_train[xmv]),lag))
            cors.append(cor)

        highest_cor_indexes = np.argsort(cors)[-4:][::-1]
        hci.append(highest_cor_indexes)
        plt.plot(cors, label=f'{xmv} (max at lags {highest_cor_indexes})')

    print(f"Highest correlation indexes for {target} and xmvs (j=1-11):")
    print(hci)
    np.save(f'data/hci_{target}.npy', hci)

    plt.legend()
    plt.savefig(f"plots/cross_correlation_{target}.pdf", format='pdf', dpi=1200)
    plt.show()


# Do the same for xmeas_2 with itself (skip the 0 lags)
target = 'xmeas_2'
cors = []
for lag in range(1, max_n_lags + 1):
    cor = np.abs(np.asarray(df_train[target]) @ np.roll(np.asarray(df_train[target]),lag))
    cors.append(cor)

highest_cor_indexes = np.argsort(cors)[-4:][::-1] + 1
print(highest_cor_indexes)







