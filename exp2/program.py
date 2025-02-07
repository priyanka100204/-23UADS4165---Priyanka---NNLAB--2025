import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class MLP:
    def __init__(self, input_size, hidden_size, output_size, lr=0.1, epochs=10000):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr
        self.epochs = epochs
        
        # Initialize weights
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        
        # Initialize biases
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
    
    def forward(self, X):
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output
    
    def backward(self, X, y):
        # Compute output error
        output_error = y - self.final_output
        output_delta = output_error * sigmoid_derivative(self.final_output)
        
        # Compute hidden layer error
        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.lr * np.dot(self.hidden_output.T, output_delta)
        self.bias_output += self.lr * np.sum(output_delta, axis=0, keepdims=True)
        self.weights_input_hidden += self.lr * np.dot(X.T, hidden_delta)
        self.bias_hidden += self.lr * np.sum(hidden_delta, axis=0, keepdims=True)
    
    def train(self, X, y):
        for _ in range(self.epochs):
            self.forward(X)
            self.backward(X, y)
    
    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# Define XOR truth table
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1)
mlp.train(X, y)

# Test MLP on XOR function
predictions = mlp.predict(X)
print("Predictions for XOR:", predictions.flatten())
