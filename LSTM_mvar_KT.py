# Keras LSTM for multiple variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
df = df.copy()

# Train-val-test split (80/19/1), but all dividable with 512 (chosen as max batch size)
train_size = int(0.8 * len(df))
train_size = train_size - (train_size % 514)
train_df = df[:train_size]

val_size = int(0.19 * len(df))
val_size = val_size - (val_size % 512)
val_df = df[train_size:train_size + val_size]

print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")

# Number of features
num_features = len(xmv_variables) + len(targets)
num_targets = len(targets)

# Hyperparameter-optimized LSTM model constuctor
def create_lstm_model(hp,input_shape,output_shape):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 2, 4)):
        model.add(LSTM(units=hp.Int('units_' + str(i), min_value=32, max_value=512, step=32),
                              return_sequences=True if i < hp.Int('num_layers', 2, 5) - 1 else False,
                              input_shape=input_shape))
        model.add(Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))
    model.add(Dense(output_shape))
    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                  loss='mse',
                  metrics=['mae'])
    return model

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

# Initialize the tuner
input_shape = (n_lags, num_features)
output_shape = num_targets
tuner = Hyperband(
    lambda hp: create_lstm_model(hp, input_shape=input_shape, output_shape=output_shape),
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

# Print the best model structure
best_trials = tuner.oracle.get_best_trials(num_trials=1)
for t in best_trials:
    print("Trial summary")
    print(t.summary())
    print("Trial hyperparameters")
    print(t.hyperparameters.values)
