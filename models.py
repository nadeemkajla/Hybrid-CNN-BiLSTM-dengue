import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv1D, Dropout, LayerNormalization,
                                     Bidirectional, LSTM, MultiHeadAttention,
                                     Add, GlobalAveragePooling1D,
                                     Concatenate, Dense)
from tensorflow.keras import Model

from config import (CONV_FILTERS, CONV_KERNEL,
                    LSTM_UNITS, NUM_HEADS, KEY_DIM)

def build_lstm_baseline(input_shape):
    x_in = Input(shape=input_shape)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(x_in)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(x_in, out, name="bilstm_baseline")

def build_hybrid_cnn_bilstm_mha(input_shape):
    x_in = Input(shape=input_shape)

    # CNN: local motif pre-tokenization
    F = Conv1D(CONV_FILTERS, CONV_KERNEL, activation="relu", padding="same")(x_in)
    F = Dropout(0.3)(F)
    F = LayerNormalization()(F)

    # BiLSTM: directional temporal memory
    H = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(F)
    H = Dropout(0.3)(H)
    H = LayerNormalization()(H)

    # MHA: post-recurrence global context over contextualized states
    Z = MultiHeadAttention(num_heads=NUM_HEADS, key_dim=KEY_DIM)(H, H)
    Z = Dropout(0.3)(Z)
    Z = Add()([H, Z])        # local residual is fine
    Z = LayerNormalization()(Z)

    # Residual fusion of pooled paths (acts like learnable gammas)
    poolZ = GlobalAveragePooling1D()(Z)
    poolH = GlobalAveragePooling1D()(H)
    poolF = GlobalAveragePooling1D()(F)
    fused = Concatenate()([poolZ, poolH, poolF])
    fused = LayerNormalization()(fused)
    fused = Dense(64, activation="relu")(fused)
    fused = Dropout(0.3)(fused)

    out = Dense(1, activation="sigmoid")(fused)
    return Model(x_in, out, name="hybrid_cnn_bilstm_mha")
