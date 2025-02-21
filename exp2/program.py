import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, epochs=100):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros(output_size)
        self.lr = learning_rate
        self.epochs = epochs

    def step_function(self, x):
        """Step activation function"""
        return np.where(x >= 0, 1, 0)

    def forward(self, X):
        """Forward pass"""
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.step_function(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.step_function(self.z2)
        return self.a2

    def backward(self, X, y, output):
        """Backward pass using weight update rule"""
        error = y - output  # Compute error

        # Adjust weights using perceptron learning rule
        self.W2 += self.lr * np.dot(self.a1.T, error)
        self.b2 += self.lr * np.sum(error, axis=0)
        self.W1 += self.lr * np.dot(X.T, np.dot(error, self.W2.T))
        self.b1 += self.lr * np.sum(np.dot(error, self.W2.T), axis=0)

    def train(self, X, y):
        """Train the MLP"""
        for epoch in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            loss = np.mean((y - output) ** 2)  # Mean Squared Error
            acc = self.accuracy(X, y)
            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")

    def predict(self, X):
        """Make predictions"""
        return self.forward(X)

    def accuracy(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        correct = np.sum(predictions == y)
        return correct / len(y) * 100  # Accuracy in percentage
# XOR Dataset
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]])  # XOR Truth Table
# Train MLP
mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1, epochs=100)
mlp.train(X_xor, y_xor)

# Test Predictions
print("\nXOR Predictions:")
for x in X_xor:
    print(f"Input: {x}, Output: {mlp.predict([x])[0]}")

# Final Accuracy
accuracy = mlp.accuracy(X_xor, y_xor)
print(f"\nFinal Model Accuracy: {accuracy:.2f}%")
