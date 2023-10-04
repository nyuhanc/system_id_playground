# Keras LSTM for multiple variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error

# Check if GPU is available
print(tf.config.experimental.list_physical_devices('GPU')) # If empty, GPU is not available

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

# Number of features
num_features = len(xmv_variables) + len(targets)
num_targets = len(targets)

# --------- Define LSTM model constructor -------------
def create_lstm_model(input_shape, output_shape, lstm_layers=[50, 50], dropout_rate=0.1, stateful=False, batch_size=None):

    inputs = Input(batch_shape=(batch_size, input_shape[0], input_shape[1]))

    x = inputs  # (batch_size, seq_len, num_features)
    for i, units in enumerate(lstm_layers):
        return_seq = True if i < len(lstm_layers) - 1 else False  # Only the last layer should return_sequences=False
        x = LSTM(units, return_sequences=return_seq, stateful=stateful)(x)
        x = Dropout(dropout_rate)(x)

    # Output Layer for Regression
    outputs = Dense(output_shape)(x)

    return Model(inputs=inputs, outputs=outputs)

# -------------------------------------------

# Define predictors: - xmvs from time t-n_lags to t (!!!)
#                    - xmeas from time t-(n_lags+1) to t-1
predictors = []
for target in targets:
    predictors.extend([f"{target}_lag{i}" for i in range(1, n_lags + 1)])
for var in xmv_variables:
    predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

# Reshape data to fit LSTM model (samples, timesteps, features)
X_train = train_df[predictors].values.reshape(-1, n_lags, num_features)
y_train = train_df[targets].values.reshape(-1 ,num_targets)
X_val = val_df[predictors].values.reshape(-1, n_lags, num_features)
y_val = val_df[targets].values.reshape(-1, num_targets)
X_test = test_df[predictors].values.reshape(-1, n_lags, num_features)
y_test = test_df[targets].values.reshape(-1, num_targets)

# Hyperparameters for the model
input_shape = (X_train.shape[1], X_train.shape[2])  # (n_lags, features)
lstm_layers = [96, 96, 64]
dropout_rate = 0
stateful = False  # stateful LSTM !!!!!
batch_size = 32  # Must satisfy 512 % batch_size == 0 (look at the train-val-test split above)

# Create the LSTM model
model = create_lstm_model(input_shape=input_shape,
                          output_shape=num_targets,
                          lstm_layers=lstm_layers,
                          dropout_rate=dropout_rate,
                          stateful=stateful,
                          batch_size=batch_size,
                          )

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# Fit the model
history = model.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_val, y_val),
    batch_size=batch_size,  # Must satisfy 512 % batch_size == 0
    epochs=2,
    shuffle=False,  # Important! Do not shuffle the data when stateful=True
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='min')
    ]
)

# Save model
model.save(f'models/LSTM_mvar_{target}_n_{n_lags}.h5')

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
    input_tensor = input_df.to_numpy(dtype='float32').reshape(n_lags, num_features)

    # Assume input_tensor is your single sample of shape (time_steps, features).
    # Since the model expects a batch of samples (it predicts batch_size samples
    # in parallel), we need to add a dummy batch dimension, i.e. convert the input
    # tensor from shape (time_steps, features) to (batch_size, time_steps, features)
    # by duplicating the sample along the batch dimension. We are duplicating the
    # sample instead of adding zeros so that the inner states of the LSTM will
    # remain the same as if we predicted one sample without a batch.
    real_batch = np.repeat(input_tensor[np.newaxis, ...], repeats=batch_size, axis=0)

    # Make a prediction
    prediction = model.predict(real_batch)[0]  # [0] to get scalar instead of array of size 1
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
n_rows = int(np.ceil(num_targets / 3.0))
fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), sharex=True)
fig.suptitle(f'n_lags={n_lags}, multitarget')

# Flatten the axes to make indexing easier
axes = axes.flatten()

# Hide any extra subplot axes that are not needed
for i in range(num_targets, n_rows * 3):
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
plt.savefig(f'plots/LSTM_mvar_n_{n_lags}.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)

# Show the plot
plt.show()
