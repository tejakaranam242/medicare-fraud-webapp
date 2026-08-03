import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization

def create_autoencoder(input_dim):
    """
    Deep Autoencoder for anomaly detection.
    Encoder compresses to a bottleneck; decoder reconstructs.
    High reconstruction error = anomaly (potential fraud).
    Bottleneck: input_dim → 64 → 32 → 16 → 32 → 64 → input_dim
    """
    inputs = Input(shape=(input_dim,))
    
    # Encoder
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    bottleneck = Dense(16, activation='relu', name='bottleneck')(x)
    
    # Decoder
    x = Dense(32, activation='relu')(bottleneck)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    outputs = Dense(input_dim, activation='linear')(x)
    
    model = Model(inputs, outputs, name='Autoencoder')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model
