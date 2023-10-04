# Description: XGBoost model for 1 variable

import numpy as np
import pandas as pd
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

# Normalized Root Mean Squared Error (with STD)
def NRMSE(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred)) / np.std(y_true)

# Parameters
n_lags = 10
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
df_train = df_train.copy()

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

    # Define predictors: - xmvs from time t-n_lags to t (!!!)
    #                    - xmeas from time t-(n_lags+1) to t-1
    predictors = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
    for var in xmv_variables:
        predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

    X_train = train_df[predictors].values
    y_train = train_df[target].values
    X_val = val_df[predictors].values
    y_val = val_df[target].values
    X_test = test_df[predictors].values
    y_test = test_df[target].values

    # Create XGBoost model
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=400,
        booster='gbtree',
        learning_rate=0.05,
        subsample=1,
        colsample_bytree=1,
        # reg_lambda=0.7530702981035536,
        # reg_alpha=0.017560230153432665,
        max_depth=8,
        # min_child_weight=9,
        # eta=3.6835373674756498e-06,
        # gamma=4.222259824436734e-06,
        # grow_policy='lossguide',
        verbosity=3,
        device='cpu',  # gpu often runs out of memory
    )

    # Train XGBoost model
    model.fit(X_train, y_train)

    # Save model
    model.save_model(f'models/XGB_1var_{target}_n_lags_{n_lags}.json')

    # Score the model on validation set
    print(f"NRMSE on val. set: "
          f"{NRMSE(model.predict(X_val), y_val)}")

    # Recursive forecasting
    predictions = []
    xmeas_lags = test_df[[f"{target}_lag{i}" for i in range(1, n_lags + 1)]].iloc[0].tolist()[::-1]
    predict_n_steps = len(y_test)

    for i in range(predict_n_steps):
        # Prepare input data for prediction
        input_data = {}
        for var in [target] + xmv_variables:
            for lag in range(0, n_lags):
                if var == target:
                    input_data[f"{var}_lag{lag + 1}"] = xmeas_lags[-(lag + 1)]
                else:
                    input_data[f"{var}_lag{lag}"] = test_df[f"{var}_lag{lag}"].iloc[i]
        input_df = pd.DataFrame([input_data])

        # Make a prediction
        prediction = model.predict(input_df)[0]
        predictions.append(prediction)

        # Update xmeas_1 lags with the new prediction
        xmeas_lags.append(prediction)

    # Calculate NRMSE  ( [:predict_n_steps] because if predict_n_steps < len(y_test) )
    nrmse_value = NRMSE(y_test[:predict_n_steps], predictions)

    # Plot predictions vs. actuals
    plot_samples = 150
    actual_values = y_test[:plot_samples]
    predicted_values = predictions[:plot_samples]

    # Plot actual and predicted values
    ax.plot(np.arange(len(predicted_values)), actual_values, label='Actual')
    ax.plot(np.arange(len(predicted_values)), predicted_values, label='Predicted')
    ax.set_title(f'n_lags={n_lags}, target={target}')

    # Add grid, legend, and NRMSE text
    ax.grid(True)
    ax.legend(loc=1)
    ax.text(0.985, 0.02, f'NRMSE: {nrmse_value:.4f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# Save the plot
plt.savefig(f'plots/XGB_1var_n_{n_lags}.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)

# Show the plot
plt.show()