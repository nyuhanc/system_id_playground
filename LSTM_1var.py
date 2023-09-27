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

# Train-val-test split (80/20/20); but a multiple of max batch size (512)
train0_size = int(0.8 * len(df))
train0_size = train0_size - (train0_size % 512)
# Train-val-test split (80/20/20)
train_size = int(0.8 * train0_size)
train_size = train_size - (train_size % 512)
train_df, val_df, test_df = df[:train_size], df[train_size:train0_size], df[train0_size:(len(df) // 512) * 512]

# Number of features
num_features = len(xmv_variables) + 1 # 11 xmv variables + 1 target variable


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
    dropout_rate = 0.1
    stateful = True  # stateful LSTM !!!!!
    batch_size = 32  # Specify batch size for stateful LSTM

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
        x=X_train, #
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=32,  # Must satisfy 512 % batch_size == 0
        epochs=5,
        shuffle=False,  # Important! Do not shuffle the data when stateful=True
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='min')
        ]
    )

    # # Score the model
    # y_pred = model.predict(X_test)
    # y_pred = y_pred.reshape(-1)
    # y_test = y_test.values
    # print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

    # Before entering the loop for recursive forecasting, reset the LSTM internal state
    model.reset_states()

    # Recursive forecasting
    predictions = []
    xmeas_lags = y_test[:n_lags].tolist()

    predict_n_steps = 128

    for i in range(predict_n_steps):  # len(y_test) - n_lags):
        # Prepare input data for prediction
        input_data = {f"{target}_lag{j + 1}": xmeas_lags[-(j + 1)] for j in range(n_lags)}
        for var in xmv_variables:
            for lag in range(0, n_lags):
                input_data[f"{var}_lag{lag}"] = test_df.iloc[i + n_lags - lag][var]
        input_df = pd.DataFrame([input_data])
        input_df = input_df.to_numpy(dtype='float32').reshape(n_lags, num_features)

        # Make a prediction
        prediction = model.predict(input_df[np.newaxis, ...])[0]
        predictions.append(prediction)

        # Update xmeas_1 lags with the new prediction
        xmeas_lags.append(prediction)

    # Calculate RMSE
    rmse_value = mean_squared_error(y_test[n_lags:n_lags + predict_n_steps], predictions, squared=False)

    # Plot predictions vs. actuals
    plot_samples = predict_n_steps
    actual_values = y_test[n_lags:n_lags + plot_samples]
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
    ax.text(0.985, 0.02, f'RMSE: {rmse_value:.4f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Save the plot
    plt.savefig(f'plots/LSTM_1var_{target}_n_lags_{n_lags}.pdf', format='pdf', dpi=1200, bbox_inches='tight',
                pad_inches=0.1)

    # Show the plot
    plt.show()







