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

# Parameters
n_lags = 60
# Multiple targets
targets = [f'xmeas_{i}' for i in range(1, 20+1)] # ['xmeas_17','xmeas_18']

# Generate lagged features for target
df_train = normalize_data(df_train_OG.copy())
for target in targets:
    for lag in range(1, n_lags + 1):
        df_train[f"{target}_lag{lag}"] = df_train[target].shift(lag)

# Generate lagged features for all xmv_j
xmv_variables = [col for col in df_train.columns if 'xmv' in col] # xmvs 1-11

for var in xmv_variables:
    for lag in range(1, n_lags + 1):
        df_train[f"{var}_lag{lag}"] = df_train[var].shift(lag)

# Drop missing values (due to lagged features creation)
df = df_train.dropna()

# Defragment the dataframe
df_train = df_train.copy()

# Train-test split (80/20)
train_size = int(0.8 * len(df))
train_df, test_df = df[:train_size], df[train_size:]


# Define predictors: xmvs from time t-n_lags to t (!!!), and xmeas from time t-n_lags to t-1
predictors = []
for target in targets:
    predictors.extend([f"{target}_lag{i}" for i in range(1, n_lags + 1)])
for var in xmv_variables:
    predictors.extend([var] + [f"{var}_lag{i}" for i in range(1, n_lags + 1)])

X_train = train_df[predictors]
y_train = train_df[targets]
X_test = test_df[predictors]
y_test = test_df[targets]

# Create XGBoost model
model = xgb.XGBRegressor(
    # n_estimators=300,
    # learning_rate=0.2,
    # objective='reg:squarederror',
    # max_depth=8,
    # max_leaves=None,
    # max_bin=None,
    # grow_policy=None,
    # verbosity=3,
    # booster=None,
    # tree_method=None,
    # gamma=None,
    # min_child_weight=None,
    # max_delta_step=None,
    # subsample=None,
    # sampling_method=None,
    # colsample_bytree=0.9,
    # colsample_bylevel=None,
    # colsample_bynode=None,
    # reg_alpha=None,
    # reg_lambda=None,
    # scale_pos_weight=None,
    # base_score=None,
    # num_parallel_tree=None,
    # random_state=None,
    # n_jobs=None,
    # monotone_constraints=None,
    # interaction_constraints=None,
    # importance_type=None,
    # gpu_id=None,
    # validate_parameters=None,
    # predictor=None,
    # enable_categorical=False,
    # feature_types=None,
    # max_cat_to_onehot=None,
    # max_cat_threshold=None,
    # eval_metric=None,
    # early_stopping_rounds=None,
    # callbacks=None,
)

# Train XGBoost model
model.fit(X_train, y_train)

# Recursive forecasting
predictions = []
xmeas_lags = {target: y_test[target].iloc[:n_lags].tolist() for target in targets}

predict_n_steps = 100

for i in range(predict_n_steps):
    # Prepare input data for prediction
    input_data = {}
    for target in targets:
        input_data.update({f"{target}_lag{j + 1}": xmeas_lags[target][-(j + 1)] for j in range(n_lags)})
    for var in xmv_variables:
        input_data[var] = test_df.iloc[i + n_lags][var]
        for lag in range(1, n_lags + 1):
            input_data[f"{var}_lag{lag}"] = test_df.iloc[i + n_lags - lag][var]

    input_df = pd.DataFrame([input_data])

    # Make a prediction
    prediction = model.predict(input_df)[0]
    predictions.append(prediction)

    # Update xmeas lags with the new prediction
    for idx, target in enumerate(targets):
        xmeas_lags[target].append(prediction[idx])

# Initialize a dictionary to hold RMSE for each target
rmse_values = {}

# Initialize a dictionary to hold predictions for each target
predictions_dict = {target: [] for target in targets}

# Separate out the predictions for each target
for prediction in predictions:
    for idx, target in enumerate(targets):
        predictions_dict[target].append(prediction[idx])

# Calculate and print RMSE for each target
for target in targets:
    rmse_value = mean_squared_error(
        y_test[target].iloc[n_lags:n_lags + predict_n_steps],
        predictions_dict[target],
        squared=False
    )
    rmse_values[target] = rmse_value
    print(f'RMSE for {target}: {rmse_value:.4f}')


# Plotting
# Determine the number of rows needed
n_rows = int(np.ceil(len(targets) / 3.0))
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), sharex=True)
fig.suptitle(f'n_lags={n_lags}(+1), multiple targets')

# Flatten the axes to make indexing easier
axes = axes.flatten()

# Hide any extra subplot axes that are not needed
for i in range(len(targets), n_rows * 3):
    axes[i].axis('off')

for idx, target in enumerate(targets):
    ax = axes[idx]

    # Plot actual and predicted values
    actual_values = y_test[target][n_lags:n_lags + predict_n_steps]
    predicted_values = predictions_dict[target][:predict_n_steps]

    ax.plot(np.arange(len(predicted_values)), actual_values, label='Actual')
    ax.plot(np.arange(len(predicted_values)), predicted_values, label='Predicted')
    ax.set_title(f'n_lags={n_lags}(+1), target={target}')

    # Add grid, legend, and RMSE text
    ax.grid(True)
    ax.legend(loc=1)
    ax.text(0.985, 0.02, f'RMSE: {rmse_values[target]:.4f}',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))


# Save the plot
plt.savefig(f'plots/xgb_multivar_{str(targets).replace(" ", "")}_n_lags_{n_lags}.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)

# Show the plot
plt.show()
