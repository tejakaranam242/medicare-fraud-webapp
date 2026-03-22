import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization
import tensorflow_addons as tfa

def create_cnn(input_shape):
    model = Sequential([
        # Using a smaller kernel and fewer layers for small feature sets
        Conv1D(64, kernel_size=2, activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(128, kernel_size=2, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss=tfa.losses.SigmoidFocalCrossEntropy(),
        metrics=['accuracy']
    )
    return model
