import numpy as np

class Perceptron:
    def __init__(self, input_size, lr=0.1, epochs=100):
        self.weights = np.random.rand(input_size + 1)  # Including bias weight
        self.lr = lr
        self.epochs = epochs
    
    def activation(self, x):
        return 1 if x >= 0 else 0
    
    def predict(self, x):
        x = np.insert(x, 0, 1)  # Adding bias input
        return self.activation(np.dot(self.weights, x))
    
    def train(self, X, y):
        for _ in range(self.epochs):
            for i in range(len(X)):
                x_i = np.insert(X[i], 0, 1)  # Adding bias input
                y_pred = self.activation(np.dot(self.weights, x_i))
                error = y[i] - y_pred
                self.weights += self.lr * error * x_i  # Update rule
    
    def evaluate(self, X, y):
        correct = sum(self.predict(x) == y_true for x, y_true in zip(X, y))
        accuracy = correct / len(y)
        return accuracy

# Define NAND and XOR truth tables
nand_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_y = np.array([1, 1, 1, 0])

xor_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_y = np.array([0, 1, 1, 0])

# Train Perceptron for NAND
nand_perceptron = Perceptron(input_size=2)
nand_perceptron.train(nand_X, nand_y)
nand_accuracy = nand_perceptron.evaluate(nand_X, nand_y)
print("NAND Perceptron Accuracy:", nand_accuracy)

# Train Perceptron for XOR
xor_perceptron = Perceptron(input_size=2)
xor_perceptron.train(xor_X, xor_y)
xor_accuracy = xor_perceptron.evaluate(xor_X, xor_y)
print("XOR Perceptron Accuracy:", xor_accuracy)