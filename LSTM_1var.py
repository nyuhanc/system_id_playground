# Keras LSTM for 1 variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Import Keras (transformer model and layers)
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LayerNormalization, LSTM, Add

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
    return np.sqrt(np.mean((y_true - y_pred)**2)) / np.std(y_true)

# Parameters
n_lags = 10
targets = ['xmeas_1'] #[f'xmeas_{i}' for i in range(1, 41+1)]

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
num_features = len(xmv_variables) + 1  # 11 xmv variables + 1 target variable

# --------- Define LSTM model constructor -------------
def create_lstm_model(input_shape, lstm_layers=[50, 50], dropout_rate=0.1, stateful=False, batch_size=None):
    inputs = Input(batch_shape=(batch_size, input_shape[0], input_shape[1]) if stateful else (
    None, input_shape[0], input_shape[1]))

    x = inputs  # (batch_size, seq_len, num_features)
    for i, units in enumerate(lstm_layers):
        return_seq = True if i < len(lstm_layers) - 1 else False  # Only the last layer should return_sequences=False
        x = LSTM(units, return_sequences=return_seq, stateful=stateful)(x)
        x = Dropout(dropout_rate)(x)

    # Output Layer for Regression
    outputs = Dense(1)(x)

    return Model(inputs=inputs, outputs=outputs)

# -----------------------------------------------------------

for target in targets:

    # Define predictors: xmvs from time t-n_lags to t (!!!), and xmeas from time t-n_lags to t-1
    predictors = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
    for var in xmv_variables:
        predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

    X_train = train_df[predictors].values.reshape(-1, n_lags, num_features)
    y_train = train_df[target].values
    X_val = val_df[predictors].values.reshape(-1, n_lags, num_features)
    y_val = val_df[target].values
    X_test = test_df[predictors].values.reshape(-1, n_lags, num_features)
    y_test = test_df[target].values

    # Hyperparameters for the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    lstm_layers = [50, 50, 50]  # Three LSTM layers with 50 units each
    dropout_rate = 0.05
    stateful = True  # stateful LSTM !!!!!
    batch_size = 32  # Must satisfy 512 % batch_size == 0 (look at the train-val-test split above)

    # Create the LSTM model
    model = create_lstm_model(input_shape=input_shape,
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
        epochs=100,
        shuffle=False,  # Important! Do not shuffle the data when stateful=True
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='min')
        ]
    )

    # Score the model on validation set
    print(f"NRMSE on val. set: "
          f"{NRMSE(model.predict(X_val).reshape(-1), y_val)}")

    # Recursive forecasting
    predictions = []
    xmeas_lags = test_df[[f"{target}_lag{i}" for i in range(1, n_lags + 1)]].iloc[0].tolist()[::-1]
    predict_n_steps = len(y_test)

    for i in range(predict_n_steps):
        # Prepare input data for prediction
        input_data = {f"{target}_lag{j + 1}": xmeas_lags[-(j + 1)] for j in range(n_lags)}
        for var in xmv_variables:
            for lag in range(0, n_lags):
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

        # Update xmeas_1 lags with the new prediction
        xmeas_lags.append(prediction)

    # Calculate NRMSE  ( [:predict_n_steps] because if predict_n_steps < len(y_test) )
    nrmse_value = NRMSE(y_test[:predict_n_steps], predictions)
    rmse_value = mean_squared_error(y_test[:predict_n_steps], predictions, squared=False)

    # Plot predictions vs. actuals
    plot_samples = 150
    actual_values = y_test[:plot_samples]
    predicted_values = predictions[:plot_samples]

    # Create a figure and axis object
    fig, ax = plt.subplots()
    fig.suptitle(f'n_lags={n_lags}(+1), traget={target}, RMSE={rmse_value:.4f}')

    # Plot actual and predicted values
    ax.plot(np.arange(len(predicted_values)), actual_values, label='Actual')
    ax.plot(np.arange(len(predicted_values)), predicted_values, label='Predicted')

    # Add grid, legend, and RMSE text
    ax.grid(True)
    ax.legend(loc=1)
    ax.text(0.985, 0.02, f'NRMSE: {rmse_value:.4f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Save the plot
    plt.savefig(f'plots/LSTM_1var_{target}_n_lags_{n_lags}.pdf', format='pdf', dpi=1200, bbox_inches='tight',
                pad_inches=0.1)

    # Show the plot
    plt.show()
