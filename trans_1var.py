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
n_lags = 3
targets = ['xmeas_1'] #[f'xmeas_{i}' for i in range(1, 41+1)]

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

# Train-test split (80/20) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
train_size = int(0.8 * len(df))
train_df, test_df = df, df[train_size:] #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# --------- Define transformer model constructor -------------
def create_transformer_model(input_shape, num_heads, feed_forward_dim, dropout_rate=0.1, num_transformer_blocks=2):
    # :param input_shape: Shape of the input data (number of timesteps, number of features)
    # :param num_heads: Number of attention heads
    # :param feed_forward_dim: Hidden layer size in feed forward network inside transformer
    # :param dropout_rate: Dropout rate
    # :param num_transformer_blocks: Number of transformer blocks

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Get positional encoding
    pos_encoding = SinePositionEncoding(input_shape[0])(inputs)

    # Combine the inputs with the positional encoding
    x = layers.Add()([inputs, pos_encoding])

    # Transformer blocks
    for _ in range(num_transformer_blocks):
        # Multi-head self-attention
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

    return Model(inputs=[inputs, pos_encoding], outputs=outputs)

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

    # Create transformer model
    input_shape = (X_train.shape[1], 1)  # (number of time steps, number of features)

    # Hyperparameters for the transformer model
    num_heads = 2
    feed_forward_dim = 128

    # Create the model
    model = create_transformer_model(input_shape, num_heads, feed_forward_dim)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # Fit the model
    history = model.fit(
        x=[X_train.values[..., np.newaxis], X_train.values[..., np.newaxis]], #
        y=y_train.values,
        validation_split=0.2,
        batch_size=32,
        epochs=100,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, mode='min')
        ]
    )

    # Evaluate the model
    y_pred = model.predict([X_test.values[..., np.newaxis], X_test.values[..., np.newaxis]])
    y_pred = y_pred.reshape(-1)
    y_test = y_test.values

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f'RMSE for {target}: {rmse:.4f}')

    # Plot loss
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title(f'Loss for {target}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    # Save the model
    model.save(f'models/transformer_{target}.keras')






