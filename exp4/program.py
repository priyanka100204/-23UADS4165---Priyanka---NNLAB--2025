import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize the data (scale pixel values to range 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Hyperparameters
activation_functions = ['sigmoid', 'relu', 'tanh']
neurons_options = [256, 128, 64]
learning_rate = 0.1
batch_size = 10
epochs = 50

# Loop through all combinations
for activation in activation_functions:
    for neurons in neurons_options:
        print(f"Training model with activation={activation}, neurons={neurons}")
        
        # Define the neural network model
        model = keras.Sequential([
            layers.Flatten(input_shape=(28, 28)),  # Input layer (flatten 28x28 images to 1D)
            layers.Dense(neurons, activation=activation),  # Hidden layer
            layers.Dense(10, activation='softmax')  # Output layer with 10 classes
        ])
        
        # Compile the model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        
        # Train the model
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2)
        
        # Evaluate the model
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        print(f"Test accuracy for activation={activation}, neurons={neurons}: {test_acc:.4f}\n")
