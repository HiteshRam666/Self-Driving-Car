import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

class CustomScalingLayer(layers.Layer):
    """Custom layer to apply scaling using atan."""
    def call(self, inputs):
        return tf.multiply(tf.atan(inputs), 2.0)

def build_model(input_shape):
    # Define the input
    inputs = tf.keras.Input(shape=input_shape)

    # First Convolutional Layer
    x = layers.Conv2D(24, (5, 5), strides=2, activation='relu')(inputs)

    # Second Convolutional Layer
    x = layers.Conv2D(36, (5, 5), strides=2, activation='relu')(x)

    # Third Convolutional Layer
    x = layers.Conv2D(48, (5, 5), strides=2, activation='relu')(x)

    # Fourth Convolutional Layer
    x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)

    # Fifth Convolutional Layer
    x = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(x)

    # Flatten Layer
    x = layers.Flatten()(x)

    # Fully Connected Layer 1
    x = layers.Dense(1164, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Fully Connected Layer 2
    x = layers.Dense(100, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Fully Connected Layer 3
    x = layers.Dense(50, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Fully Connected Layer 4
    x = layers.Dense(10, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output Layer
    x = layers.Dense(1, activation='tanh')(x)

    # Apply custom scaling
    outputs = CustomScalingLayer()(x)

    return Model(inputs=inputs, outputs=outputs)

# Build the model
input_shape = (66, 200, 3)  # Consistent with the image size from `driving_data.py`
model = build_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mse')  # Using Mean Squared Error for steering angle prediction

# Print Model Summary
model.summary()
