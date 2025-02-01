import numpy as np

# Define the Perceptron class
class Perceptron:
    def __init__(self, input_size, learning_rate=0.01, epochs=1000):
        # Initialize weights and bias
        self.weights = np.zeros(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    # Activation function: Step function
    def activation(self, z):
        return 1 if z >= 0 else 0

    # Predict output for a given input
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation(linear_output)
        return  y_predicted

    # Train the Perceptron
    def train(self, X, y):
        n_samples, n_features = X.shape
        for _ in range(self.epochs):
            for i in range(len(X)):
                # Calculate the prediction
                prediction = self.predict(X[i])
                # Update weights and bias based on the error
                error = y[i] - prediction
                self.weights += self.learning_rate * error * X[i]
                self.bias += self.learning_rate * error

# Example usage
if __name__ == "__main__":
    # Input data (OR Gate Example)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Target output (OR Gate Result)
    y = np.array([0, 1, 1, 1])

    # Initialize Perceptron with 2 inputs (since X has 2 features)
    perceptron = Perceptron(input_size=2)

    # Train the Perceptron
    perceptron.train(X, y)

    #weights
    print("weights", perceptron.weights)
    # Test the trained Perceptron
    print("Test predictions:")
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Predicted Output: {perceptron.predict(X[i])}")
