# Description: XGBoost model for 1 variable but with lags found to be most correlated with target

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load data
df_train_OG = pd.read_csv('../data/faultfreetraining.txt')
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
target = 'xmeas_2'
# Load laggs
lags_xmv = np.load('../data/hci_xmeas_2.npy')
lags_target = np.asarray([1, 6, 7, 2])

# Generate lagged features for target
df_train = normalize_data(df_train_OG.copy())
for lag in lags_target:
    df_train[f"{target}_lag{lag}"] = df_train[target].shift(lag)

# Generate lagged features for all xmv_j
xmv_variables = [col for col in df_train.columns if 'xmv' in col] # xmvs 1-11

for i,var in enumerate(xmv_variables):
    for lag in lags_xmv[i]:
        df_train[f"{var}_lag{lag}"] = df_train[var].shift(lag)

# Drop missing values (due to lagged features creation)
df = df_train.dropna()

# Defragment the dataframe
df_train = df_train.copy()

# Train-test split (80/20)
train_size = int(0.8 * len(df))
train_df, test_df = df[:train_size], df[train_size:]


# Define predictors: xmvs from time t-n_lags to t (!!!), and xmeas from time t-n_lags to t-1
predictors = [f"{target}_lag{i}" for i in lags_target]
for i,var in enumerate(xmv_variables):
    predictors.extend([f"{var}_lag{i}" for i in lags_xmv[i]])

X_train = train_df[predictors]
y_train = train_df[target]
X_test = test_df[predictors]
y_test = test_df[target]

# Create XGBoost model
# Create XGBoost model
model = xgb.XGBRegressor(
    n_estimators=300,
    learning_rate=0.2,
    objective='reg:squarederror',
    max_depth=8,
    max_leaves=None,
    max_bin=None,
    grow_policy=None,
    verbosity=3,
    booster=None,
    tree_method=None,
    gamma=1, # min_split_loss
    min_child_weight=3,
    max_delta_step=None,
    subsample=None,
    sampling_method=None,
    colsample_bytree=0.9,
    colsample_bylevel=None,
    colsample_bynode=None,
    reg_alpha=None, #0.1,
    reg_lambda=None,#10,
    scale_pos_weight=None,
    base_score=None,
    num_parallel_tree=None,
    random_state=None,
    n_jobs=None,
    monotone_constraints=None,
    interaction_constraints=None,
    importance_type=None,
    gpu_id=None,
    validate_parameters=None,
    predictor=None,
    enable_categorical=False,
    feature_types=None,
    max_cat_to_onehot=None,
    max_cat_threshold=None,
    eval_metric=None,
    early_stopping_rounds=None,
    callbacks=None,
)

# Train XGBoost model
model.fit(X_train, y_train)

# Score the model
score = model.score(X_test, y_test)
print(score)

# Recursive forecasting
predictions = []
xmeas_lags = y_test.iloc[:np.max(lags_target)+1].tolist()

predict_n_steps = 100

for i in range(predict_n_steps): # len(y_test) - n_lags):
    # Prepare input data for prediction
    input_data = {f"{target}_lag{j}": xmeas_lags[-(j+1)] for j in lags_target}
    for i,var in enumerate(xmv_variables):
        for lag in lags_xmv[i]:
            input_data[f"{var}_lag{lag}"] = test_df.iloc[i + np.max(lags_target) - lag][var]
    input_df = pd.DataFrame([input_data])

    # Make a prediction
    prediction = model.predict(input_df)[0]
    predictions.append(prediction)

    # Update xmeas_1 lags with the new prediction
    xmeas_lags.append(prediction)

# Calculate RMSE
rmse_value = mean_squared_error(y_test.iloc[np.max(lags_target):np.max(lags_target)+predict_n_steps], predictions, squared=False)

# Plot predictions vs. actuals
plot_samples = predict_n_steps
actual_values = y_test[np.max(lags_target):np.max(lags_target)+plot_samples]
predicted_values = predictions[:plot_samples]

# Create a figure and axis object
fig, ax = plt.subplots()
fig.suptitle(f'n_lags={np.max(lags_target)}, traget={target}, RMSE={rmse_value:.4f}')

# Plot actual and predicted values
ax.plot(np.arange(len(predicted_values)), actual_values, label='Actual')
ax.plot(np.arange(len(predicted_values)), predicted_values, label='Predicted')

# Add grid, legend, and RMSE text
ax.grid(True)
ax.legend(loc=1)
ax.text(0.985, 0.02, f'RMSE: {rmse_value:.4f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

# Save the plot
plt.savefig(f'plots/xgb_1var_{target:02}_n_{np.max(lags_target)}.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)

# Show the plot
plt.show()

