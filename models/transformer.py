import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, Flatten, MultiHeadAttention,
    LayerNormalization, Dropout, Add, GlobalAveragePooling1D
)

def create_transformer(input_shape):
    """
    Enterprise Transformer Encoder block with residual connections.
    - Positional embedding via Conv1D
    - 2 transformer encoder blocks with residual connections
    - Global average pooling for sequence aggregation
    """
    inputs = Input(shape=input_shape)
    
    # Positional encoding via 1D convolution (learns local patterns)
    x = Conv1D(64, kernel_size=1, activation='relu', padding='same')(inputs)
    
    # Transformer Encoder Block 1
    attn_out1 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = Add()([x, attn_out1])                 # Residual connection
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward sub-layer
    ff1 = Dense(128, activation='relu')(x)
    ff1 = Dropout(0.2)(ff1)
    ff1 = Dense(64)(ff1)
    x = Add()([x, ff1])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Transformer Encoder Block 2
    attn_out2 = MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = Add()([x, attn_out2])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    ff2 = Dense(128, activation='relu')(x)
    ff2 = Dropout(0.2)(ff2)
    ff2 = Dense(64)(ff2)
    x = Add()([x, ff2])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Aggregation
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return model
