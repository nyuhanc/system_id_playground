import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error

from scipy.signal import impulse, lti, step, butter, lfilter, welch
from scipy.fft import fft, ifft, fftshift, fftfreq

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
plt.figure(figsize=(12, 10))
sns.heatmap(df_train[[f'xmeas_{j}' for j in range(1, 41+1)]+[f'xmv_{j}' for j in range(1, 11+1)]].corr())
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig('plots/cross_correlation_matrix.pdf', format='pdf', dpi=1200)
plt.show()

# Compute cross-correltaion between xmv variables and lagged xmeas variables
# For each xmv variable,compute cross correlation with lagged xmeas variables, each for at max 10 lags
# For each xmv variable, find the lag with the highest cross correlation

# Parameters
max_n_lags = 100 # max number of lags to consider
targets = [f'xmeas_{i}' for i in range(1, 21+1)]
xmv_variables = [f'xmv_{i}' for i in range(1, 11+1)]
fs = 1 # define sampling frequency to be 1
freqs = np.fft.fftfreq(len(df_train['xmeas_1']), 1)

# For each xmeas, compute cross correlation with each xmv for each lag
for target in targets:

    # Plot cross correlation
    plt.figure(figsize=(6,5))
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
    plt.tight_layout()
    np.save(f'data/hci_{target}.npy', hci)

    plt.legend()
    plt.savefig(f"plots/cross_correlation_{target}.pdf", format='pdf', dpi=1200)
    plt.show()

    plt.gcf().clear() # clear figure

    # Plot the fourier transform of each signal
    fig, axn = plt.subplots(1, 3, figsize=(12,2.5))
    fig.suptitle(target)
    f, psd_u = welch(df_train[target], fs, nperseg=len(df_train[target]) // 2)
    ft_u = np.fft.fft(df_train[target])
    axn[0].plot(f, psd_u)
    axn[1].plot(freqs, np.real(ft_u))
    axn[2].plot(freqs, np.imag(ft_u))
    axn[0].set_title('PSD')
    axn[1].set_title('Re')
    axn[2].set_title('Im')
    axn[0].grid(True)
    axn[1].grid(True)
    axn[2].grid(True)
    plt.tight_layout()
    plt.savefig(f'plots/fft_{target}.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)
    plt.show()

# # Do the same for xmeas_2 with itself (skip the 0 lags)
# target = 'xmeas_1'
# cors = []
# for lag in range(1, max_n_lags + 1):
#     cor = np.abs(np.asarray(df_train[target]) @ np.roll(np.asarray(df_train[target]),lag))
#     cors.append(cor)
#
# highest_cor_indexes = np.argsort(cors)[-4:][::-1] + 1
# print(highest_cor_indexes)







