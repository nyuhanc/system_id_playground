# Keras NN for multiple variable output, using Keras Tuner to tune hyperparameters

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
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
train_size = train_size - (train_size % 512)
train_df = df[:train_size]

val_size = int(0.19 * len(df))
val_size = val_size - (val_size % 512)
val_df = df[train_size:train_size + val_size]

print(f"Train size: {len(train_df)}")
print(f"Val size: {len(val_df)}")

# Number of features
num_features = len(xmv_variables) + len(targets)
num_targets = len(targets)

# ---- Modified NN constructor to accept hyperparameters ------
def create_simple_NN(hp, input_shape, output_shape):
    inputs = Input(shape=input_shape)

    x = inputs  # Initialize x to be inputs for the first layer

    # Tuning the number of hidden layers and their units
    for i in range(hp.Int('num_hidden_layers', 1, 5)):
        x = Dense(hp.Int(f'hidden_units_{i}', 32, 256, step=32), activation='relu')(inputs if i == 0 else x)
        x = Dropout(hp.Float('dropout_rate', 0.0, 0.5, step=0.1))(x)

    outputs = Dense(output_shape)(x)

    # Tuning the learning rate
    optimizer = keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]))

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

# -----------------------------------------------------------

# Define predictors: - xmvs from time t-n_lags to t (!!!)
#                    - xmeas from time t-(n_lags+1) to t-1
predictors = []
for target in targets:
    predictors.extend([f"{target}_lag{i}" for i in range(1, n_lags + 1)])
for var in xmv_variables:
    predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

# Reshape data to fit LSTM model (samples, timesteps, features)
X_train = train_df[predictors].values
y_train = train_df[target].values
X_val = val_df[predictors].values
y_val = val_df[target].values

# Initialize the tuner and perform hypertuning
input_shape = X_train.shape[1]  # n_lags * num_features
output_shape = num_targets      # num_targets
tuner = Hyperband(
    lambda hp: create_simple_NN(hp, input_shape=input_shape, output_shape=output_shape),
    objective='val_loss',
    max_epochs=40,
    factor=3,
    directory='keras_tuner_dir',
    project_name=f'keras_tuner_{target}',
    overwrite=True
)

tuner.search(X_train, y_train,
             validation_data=(X_val, y_val),
             epochs=40, # Replaced by max_epochs in Hyperband
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
