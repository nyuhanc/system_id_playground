# ARX for 1 variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.basis_function._basis_function import Polynomial

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

# Normalized Root Mean Squared Error (with STD)
def NRMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred)) / np.std(y_true)

# Parameters
n_lags = 5
targets = [
    'xmeas_1',   # Very good 1var fit
    # 'xmeas_7',  # Not that good 1var fit
    # 'xmeas_10',  # Very good 1var fit
    # 'xmeas_12',  # Very good 1var fit
    # 'xmeas_13',  # Not that good 1var fit
    # 'xmeas_15',  # Very good 1var fit
    # 'xmeas_16',  # Not that good 1var fit
    # 'xmeas_17',  # Very good 1var fit
    # 'xmeas_18',  # Not that good 1var fit
    # 'xmeas_19',  # Not that good 1var fit
    # 'xmeas_20',  # Not that good 1var fit
    # 'xmeas_21',  # Not that good 1var fit
]
xmv_variables = [col for col in df_train_OG.columns if 'xmv' in col] # xmvs 1-11

# Generate lagged features for target
df_train = normalize_data(df_train_OG.copy())

# This is due to correlation between xmv's and xmeas's at same time step (property of the data)
df_train['xmeas_1'] = df_train['xmeas_1'].shift(-1)

# Drop missing values (due to lagged features creation)
df = df_train.dropna()

# Defragment the dataframe
df = df_train.copy()

# Train-val-test split (80/19/1), but all dividable with 512 (chosen as max batch size)
train_size = int(0.8 * len(df))
train_size = train_size - (train_size % 512)
train_df = df[:train_size]

val_size = int(0.19 * len(df))
val_size = val_size - (val_size % 512)
val_df = df[train_size:train_size + val_size]

test_size = len(df) - train_size - val_size
test_size = test_size - (test_size % 512)
test_df = df[train_size + val_size:train_size + val_size + test_size]

print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")
print(f"Test size: {len(test_df)}")

# ---- Prepare for plotting results ----
# Create a figure and axis array (3 columns, with enough rows to hold all targets)
n_cols = len(targets) if len(targets) < 3 else 3
n_rows = int(np.ceil(len(targets) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=True)
fig.suptitle(f'n_lags={n_lags}, 1var models')
if len(targets) != 1:
    axes = axes.flatten()
    for i in range(len(targets), n_rows * n_cols):  # Hide unused axes
        axes[i].axis('off')
else:
    axes = [axes]

# ---- Main loop ----
for idx, target in enumerate(targets):
    ax = axes[idx]

    X_train = train_df[xmv_variables].values
    y_train = train_df[target].values.reshape(-1, 1)
    X_val = val_df[xmv_variables].values
    y_val = val_df[target].values.reshape(-1, 1)
    X_test = test_df[xmv_variables].values
    y_test = test_df[target].values.reshape(-1, 1)

    model = FROLS(
        ylag=n_lags,
        xlag = [[1 for i in range(n_lags+1)] for _ in range(len(xmv_variables))],
        elag=n_lags,
        order_selection=True,
        info_criteria='aic',  # Akaike Information Criterion
        n_info_values=6,
        estimator='least_squares',
        extended_least_squares=False,
        basis_function=Polynomial(degree=1)  # linear
    )

    model.fit(X=X_train, y=y_train)
    yhat = model.predict(X=X_val, y=y_val)

    plt.title('The ARX model cannot capture the dynamics of the system')
    plt.plot(yhat[:100, 0], 'r')
    plt.plot(y_val[:100, 0], 'b')
    plt.show()

