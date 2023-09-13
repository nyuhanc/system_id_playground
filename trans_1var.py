# Keras Transformer for 1 variable

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
from keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Add
from keras_nlp.layers import SinePositionEncoding

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
n_lags = 4
targets = ['xmeas_2'] #[f'xmeas_{i}' for i in range(1, 41+1)]

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

# Train-test split (80/20)
train_size = int(0.8 * len(df))
train_df, test_df = df[:train_size], df[train_size:]

# Number of features
num_features = len(xmv_variables) + 1 # 11 xmv variables + 1 target variable

# --------- Define transformer model constructor -------------
def create_transformer_model(input_shape, num_heads, feed_forward_dim, dropout_rate=0.1, num_transformer_blocks=2):
    # :param input_shape: Shape of the input data (batch_size, number of timesteps, number of features)
    # :param num_heads: Number of attention heads
    # :param feed_forward_dim: Hidden layer size in feed forward network inside transformer
    # :param dropout_rate: Dropout rate
    # :param num_transformer_blocks: Number of transformer blocks

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Get positional encoding
    pos_encoding = SinePositionEncoding(1000)(inputs)  # !!!!!!!!!! optimise this

    # Combine the inputs with the positional encoding
    x = layers.Add()([inputs, pos_encoding])

    for _ in range(num_transformer_blocks):
        # Multi-head self-attention: key_dim(embedding_dim)=number of features
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=input_shape[1])(x, x)
        attn_output = Dropout(dropout_rate)(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward neural network
        ffn_output = Dense(feed_forward_dim, activation='relu')(out1)
        ffn_output = Dense(input_shape[1])(ffn_output)
        ffn_output = Dropout(dropout_rate)(ffn_output)

        # Add and normalize
        x = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)

    # Output layer for regression
    # Average the outputs of all transformer blocks (can prevent overfitting, or the
    # result to be too dependent on just a few values of the output tensor)
    global_avg_pool = layers.GlobalAveragePooling1D()(x)
    outputs = Dense(1)(global_avg_pool)

    return Model(inputs=inputs, outputs=outputs)

# -----------------------------------------------------------

for target in targets:

    # Define predictors: xmvs from time t-n_lags to t (!!!), and xmeas from time t-n_lags to t-1
    predictors = [f"{target}_lag{i}" for i in range(1, n_lags + 1)]
    for var in xmv_variables:
        predictors.extend([f"{var}_lag{i}" for i in range(0, n_lags)])

    X_train = train_df[predictors].values.reshape(-1, n_lags, num_features)
    y_train = train_df[target].values
    X_test = test_df[predictors].values.reshape(-1, n_lags, num_features)
    y_test = test_df[target].values

    # Hyperparameters for the model
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_heads = 6
    feed_forward_dim = 128
    num_transformer_blocks = 4
    dropout_rate = 0.1

    # Create the model
    model = create_transformer_model(
        input_shape=input_shape,
        num_heads=num_heads,
        feed_forward_dim=feed_forward_dim,
        dropout_rate=dropout_rate,
        num_transformer_blocks=num_transformer_blocks
    )

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    # Fit the model
    history = model.fit(
        x=X_train, #
        y=y_train,
        validation_split=0.2,
        batch_size=32,
        epochs=100,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, mode='min')
        ]
    )

    # # Score the model
    # y_pred = model.predict(X_test)
    # y_pred = y_pred.reshape(-1)
    # y_test = y_test.values
    # print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}")

    # Recursive forecasting
    predictions = []
    xmeas_lags = y_test[:n_lags].tolist()

    predict_n_steps = 100

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
    plt.savefig(f'plots/trans_1var_{target:02}_n_lags_{n_lags}.pdf', format='pdf', dpi=1200, bbox_inches='tight',
                pad_inches=0.1)

    # Show the plot
    plt.show()







