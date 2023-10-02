# Keras NN for 1 variable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
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

# --------- Simple NN constructor -------------

def create_simple_NN(input_shape, hidden_layer_sizes, dropout_rate=0.1):

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Hidden layers
    for i in range(len(hidden_layer_sizes)):
        x = Dense(hidden_layer_sizes[i], activation='relu')(inputs)
        x = Dropout(dropout_rate)(x)

    # Output layer for regression
    outputs = Dense(1)(x)

    return Model(inputs=inputs, outputs=outputs)

# -----------------------------------------------------------

for target in targets:

    # Define predictors: - xmvs from time t-n_lags to t (!!!)
    #                    - xmeas from time t-(n_lags+1) to t-1
    predictors = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
    for var in xmv_variables:
        predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

    # Reshape data to fit LSTM model (samples, timesteps, features)
    X_train = train_df[predictors].values
    y_train = train_df[target].values
    X_val = val_df[predictors].values
    y_val = val_df[target].values
    X_test = test_df[predictors].values
    y_test = test_df[target].values

    # Hyperparameters for the model
    input_shape = (X_train.shape[1])  # n_lags * num_features
    hidden_layer_sizes = [192,96,256,224]
    dropout_rate = 0
    batch_size = 32  # Must satisfy 512 % batch_size == 0 (look at the train-val-test split above)

    # Create the model
    model = create_simple_NN(input_shape=input_shape,
                             hidden_layer_sizes=hidden_layer_sizes,
                             dropout_rate=dropout_rate,
                             )

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=100,
        verbose=1,
        shuffle=False,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='min')
        ]
    )

    # Save model
    model.save(f'models/NN_1var_{target}_n_lags_{n_lags}.h5')

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
        input_tensor = input_df.to_numpy(dtype='float32')

        # Make a prediction
        prediction = model.predict(input_tensor)[0]
        predictions.append(prediction)

        # Update xmeas_1 lags with the new prediction
        xmeas_lags.append(prediction)

    # Calculate NRMSE  ( [:predict_n_steps] because if predict_n_steps < len(y_test) )
    nrmse_value = NRMSE(y_test[:predict_n_steps], predictions)

    # Plot predictions vs. actuals
    plot_samples = 150
    actual_values = y_test[:plot_samples]
    predicted_values = predictions[:plot_samples]

    # Create a figure and axis object
    fig, ax = plt.subplots()
    fig.suptitle(f'NN: n={n_lags}, tar={target}')

    # Plot actual and predicted values
    ax.plot(np.arange(len(predicted_values)), actual_values, label='Actual')
    ax.plot(np.arange(len(predicted_values)), predicted_values, label='Predicted')

    # Add grid, legend, and NRMSE text
    ax.grid(True)
    ax.legend(loc=1)
    ax.text(0.985, 0.02, f'NRMSE: {nrmse_value:.4f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
            color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Save the plot
    plt.savefig(f'plots/NN_1var_{target}_n_{n_lags}.pdf', format='pdf', dpi=1200, bbox_inches='tight',
                pad_inches=0.1)

    # Show the plot
    plt.show()