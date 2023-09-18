# Keras NN for 1 variable, using Keras Tuner to tune hyperparameters

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from kerastuner.tuners import RandomSearch, Hyperband

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
targets = ['xmeas_1'] #[f'xmeas_{i}' for i in range(1, 21+1)]

# Generate lagged features for target
df_train = normalize_data(df_train_OG.copy())
for target in targets:
    for lag in range(1, n_lags + 1):
        df_train[f"{target}_lag{lag}"] = df_train[target].shift(lag)

# Generate lagged features for all xmv_j
xmv_variables = [col for col in df_train.columns if 'xmv' in col] # xmvs 1-11

for var in xmv_variables:
    for lag in range(0, n_lags + 1):
        df_train[f"{var}_lag{lag}"] = df_train[var].shift(lag)

# Drop missing values (due to lagged features creation)
df = df_train.dropna()

# Defragment the dataframe
df_train = df_train.copy()

# Train-test split (80/20)
train_size = int(0.8 * len(df))
train_df, test_df = df[:train_size], df[train_size:]

# ---- Modified NN constructor to accept hyperparameters ------
def create_simple_NN(hp, input_shape):
    inputs = Input(shape=input_shape)

    # Tuning the number of hidden layers and their units
    for i in range(hp.Int('num_hidden_layers', 1, 5)):
        x = Dense(hp.Int(f'hidden_units_{i}', 32, 256, step=32), activation='relu')(inputs if i == 0 else x)
        x = Dropout(hp.Float('dropout_rate', 0.0, 0.5, step=0.1))(x)

    outputs = Dense(1)(x)

    # Tuning the learning rate
    optimizer = keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4]))

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    return model

# -----------------------------------------------------------

for target in targets:

    # Define predictors: xmvs from time t-n_lags to t (!!!), and xmeas from time t-n_lags to t-1
    predictors = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
    for var in xmv_variables:
        predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags + 1)])

    X_train = train_df[predictors]
    y_train = train_df[target]
    X_test = test_df[predictors]
    y_test = test_df[target]


    input_shape = (X_train.shape[1],)

    # Initialize the tuner and perform hypertuning
    tuner = Hyperband(
        lambda hp: create_simple_NN(hp, input_shape=input_shape),
        objective='val_loss',
        max_epochs=40,
        factor=3,
        directory='keras_tuner_dir',
        project_name=f'keras_tuner_{target}',
        overwrite=True
    )

    tuner.search(X_train, y_train,
                 validation_split=0.2,
                 epochs=100, # Replaced by max_epochs in Hyperband
                 batch_size=32,
                 callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')],
                 verbose=2)

    # Retrieve the best model and hyperparameters
    model = tuner.get_best_models(num_models=1)[0]

    # # Score the model
    # y_pred = model.predict(X_test)
    # y_pred = y_pred.reshape(-1)
    # y_test = y_test.values
    # print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

    # Recursive forecasting
    predictions = []
    xmeas_lags = y_test.iloc[:n_lags].tolist()

    predict_n_steps = 100

    for i in range(predict_n_steps): # len(y_test) - n_lags):
        # Prepare input data for prediction
        input_data = {f"{target}_lag{j + 1}": xmeas_lags[-(j + 1)] for j in range(n_lags)}
        for var in xmv_variables:
            input_data[var] = test_df.iloc[i + n_lags][var]
            for lag in range(1, n_lags + 1):
                input_data[f"{var}_lag{lag}"] = test_df.iloc[i + n_lags - lag][var]
        input_df = pd.DataFrame([input_data])
        input_df = input_df.values.astype('float32')

        # Make a prediction
        prediction = model.predict(input_df)[0]
        predictions.append(prediction)

        # Update xmeas_1 lags with the new prediction
        xmeas_lags.append(prediction)

    # Calculate RMSE
    rmse_value = mean_squared_error(y_test.iloc[n_lags:n_lags+predict_n_steps], predictions, squared=False)

    # Plot predictions vs. actuals
    plot_samples = predict_n_steps
    actual_values = y_test[n_lags:n_lags+plot_samples]
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
    ax.text(0.985, 0.02, f'RMSE: {rmse_value:.4f}', transform=ax.transAxes, ha='right', va='bottom', fontsize=10, color='black', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))

    # Save the plot
    plt.savefig(f'plots/NN_1var_{target}_n_lags_{n_lags}.pdf', format='pdf', dpi=1200, bbox_inches='tight', pad_inches=0.1)

    # Show the plot
    plt.show()














