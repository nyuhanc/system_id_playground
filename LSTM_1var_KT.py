# Keras LSTM for 1 variable, using Keras Tuner to tune hyperparameters

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, LSTM
from kerastuner.tuners import Hyperband
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

# Parameters
n_lags = 10
targets = ['xmeas_1']

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

print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")

# Number of features
num_features = len(xmv_variables) + 1  # 11 xmv variables + 1 target variable

# Hyperparameter-optimized LSTM model constructor
def create_lstm_model(hp, input_shape):
    inputs = Input(shape=input_shape)
    x = inputs
    for i in range(hp.Int('num_lstm_layers', 1, 3)):
        x = LSTM(hp.Int(f'lstm_units_{i}', min_value=32, max_value=256, step=32),
                 return_sequences=True if i < hp.Int('num_lstm_layers', 1, 3) - 1 else False)(x)
        x = Dropout(hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1))(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(
                  hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='mse',
                  metrics=['mae'])
    return model

# -----------------------------------------------------------

for target in targets:

    # Define predictors: - xmvs from time t-n_lags to t (!!!)
    #                    - xmeas from time t-(n_lags+1) to t-1
    predictors = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
    for var in xmv_variables:
        predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

    # Reshape data to fit LSTM model (samples, timesteps, features)
    X_train = train_df[predictors].values.reshape(-1, n_lags, num_features)
    y_train = train_df[target].values
    X_val = val_df[predictors].values.reshape(-1, n_lags, num_features)
    y_val = val_df[target].values

    # Initialize the tuner
    input_shape = (X_train.shape[1], X_train.shape[2])
    tuner = Hyperband(
        lambda hp: create_lstm_model(hp, input_shape=input_shape),
        objective='val_loss',
        max_epochs=30,
        factor=3,
        directory='keras_tuner_dir',
        project_name='keras_tuner_lstm',
        overwrite=True
    )

    # Perform hyperparameter search
    tuner.search(X_train, y_train,
                 validation_data=(X_val, y_val),
                 epochs=100,
                 batch_size=32,
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')],
                 verbose=2)

    # Print the best hyperparameters
    print(f"Best hyperparameters for {target}:")
    print(tuner.get_best_hyperparameters()[0].values)
