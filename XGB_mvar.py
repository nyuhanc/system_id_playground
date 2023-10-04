# Description: XGBoost model for multiple targets

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
# Including targets with relatively good 1var fit
targets = [
    'xmeas_1',   # Very good 1var fit
    'xmeas_7',  # Not that good 1var fit
    'xmeas_10',  # Very good 1var fit
    'xmeas_12',  # Very good 1var fit
    'xmeas_13',  # Not that good 1var fit
    'xmeas_15',  # Very good 1var fit
    'xmeas_16',  # Not that good 1var fit
    'xmeas_17',  # Very good 1var fit
    'xmeas_18',  # Not that good 1var fit
    'xmeas_19',  # Not that good 1var fit
    'xmeas_20',  # Not that good 1var fit
    'xmeas_21',  # Not that good 1var fit
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

# -------------------------------------------

# Define predictors: - xmvs from time t-n_lags to t (!!!)
#                    - xmeas from time t-(n_lags+1) to t-1
predictors = []
for target in targets:
    predictors.extend([f"{target}_lag{i}" for i in range(1, n_lags + 1)])
for var in xmv_variables:
    predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

X_train = train_df[predictors].values
y_train = train_df[targets].values
X_val = val_df[predictors].values
y_val = val_df[targets].values
X_test = test_df[predictors].values
y_test = test_df[targets].values

# Create XGBoost model
model = xgb.XGBRegressor(
    colsample_bytree=0.95,
    gamma=0.0,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    n_estimators=499,
    n_jobs=-1
)

# Train XGBoost model
model.fit(X_train, y_train)

# Save model
model.save_model(f'models/XGB_mvar_n_lags_{n_lags}.json')

# Score the model on validation set
print(f"NRMSE on val. set: "
      f"{NRMSE(model.predict(X_val), y_val)}")

# Recursive forecasting
predictions = []
xmeas_lags = {target: test_df[[f"{target}_lag{i}" for i in range(1, n_lags + 1)]].iloc[0].tolist()[::-1]
              for target in targets}
predict_n_steps = len(y_test)

for i in range(predict_n_steps):
    # Prepare input data for prediction
    input_data = {}
    for var in targets + xmv_variables:
        for lag in range(0, n_lags):
            if var in targets:
                input_data[f"{var}_lag{lag + 1}"] = xmeas_lags[var][-(lag + 1)]
            else:
                input_data[f"{var}_lag{lag}"] = test_df[f"{var}_lag{lag}"].iloc[i]
    input_df = pd.DataFrame([input_data])

    # Make a prediction
    prediction = model.predict(input_df)[0]
    predictions.append(prediction)

    # Update xmeas lags with the new prediction
    for idx, target in enumerate(targets):
        xmeas_lags[target].append(prediction[idx])

# Initialize a dictionary to hold NRMSE for each target
nrmse_values = {}

# Initialize a dictionary to hold predictions for each target
predictions_dict = {target: [] for target in targets}

# Separate out the predictions for each target
for prediction in predictions:
    for idx, target in enumerate(targets):
        predictions_dict[target].append(prediction[idx])

# Calculate and print NRMSE for each target
for target in targets:
    nrmse_value = NRMSE(
        test_df[target][:predict_n_steps],
        predictions_dict[target],
    )
    nrmse_values[target] = nrmse_value
    print(f'NRMSE for {target}: {nrmse_value:.4f}')

# Plotting
# Determine the number of rows needed
n_rows = int(np.ceil(len(targets) / 3.0))
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), sharex=True)
fig.suptitle(f'n_lags={n_lags}, multitarget')

# Flatten the axes to make indexing easier
axes = axes.flatten()

# Hide any extra subplot axes that are not needed
for i in range(len(targets), n_rows * 3):
    axes[i].axis('off')

plot_samples = 150
for idx, target in enumerate(targets):
    ax = axes[idx]

    # Plot actual and predicted values
    actual_values = test_df[target][:plot_samples]
    predicted_values = predictions_dict[target][:plot_samples]

    ax.plot(np.arange(len(predicted_values)), actual_values, label='Actual')
    ax.plot(np.arange(len(predicted_values)), predicted_values, label='Predicted')
    ax.set_title(f'n_lags={n_lags}, target={target}')

    # Add grid, legend, and RMSE text
    ax.grid(True)
    ax.legend(loc=1)
    ax.text(0.985, 0.02, f'NRMSE: {nrmse_values[target]:.4f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))


# Save the plot
plt.savefig(f'plots/XGB_mvar_n_{n_lags}.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)

# Show the plot
plt.show()
