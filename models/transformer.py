import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, MultiHeadAttention, LayerNormalization

def create_transformer(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, 3, activation="relu")(inputs)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = LayerNormalization()(attn_out)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(1, activation="sigmoid")(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
