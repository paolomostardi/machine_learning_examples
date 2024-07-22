

import numpy as np

# Sigmoid activation function and its derivative
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Mean Squared Error Loss and its derivative
def mse_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def mse_loss_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

# Initialize the network parameters
input_size = 2
hidden_size = 3
output_size = 1
learning_rate = 0.1

np.random.seed(42)  # For reproducibility

# Weights and biases
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Forward propagation
def forward_propagation(X):
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Backward propagation
def backward_propagation(X, Y, Z1, A1, Z2, A2):
    m = X.shape[0]
    
    dZ2 = mse_loss_derivative(Y, A2) * sigmoid_derivative(Z2)
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m
    
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(Z1)
    dW1 = np.dot(X.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    
    return dW1, db1, dW2, db2

# Update parameters
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Training the network
def train(X, Y, epochs, learning_rate):
    global W1, b1, W2, b2
    for epoch in range(epochs):
        # Forward propagation
        Z1, A1, Z2, A2 = forward_propagation(X)
        
        # Compute loss
        loss = mse_loss(Y, A2)
        
        # Backward propagation
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2)
        
        # Update parameters
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss}')

# Example dataset (XOR problem)
training_set = [[[0,0],[0]],
                [[0,1],[0]],
                [[-1,0],[0]],
                [[0,-1],[0]],
                [[1,1],[1]],
                [[2,1],[1]], 
                [[3,1],[1]],
                [[4,1],[1]], 
                [[0.2,0.5],[0]],      
                [[0.2,0.5],[0]],
                [[0.8,0.5],[0]],
                [[0.8,0.9],[0]],
                [[1.8,0.9],[1]], 
                [[1.8,0.4],[0]],        
                [[1,0.4],[0]],    
                [[2,0.4],[0]], 
                [[3,0.4],[0]],
                [[3,0.1],[0]],
                [[3,0],[0]], 
                [[3,0.5],[1]], 
                [[2.9,0.5],[1]], 
                ]

X = [x[0] for x in training_set ]
Y = [y[1] for y in training_set ]
X = np.array(X)
Y = np.array(Y)

# Train the neural network
train(X, Y, epochs=10000, learning_rate=0.1)

# Test the trained network
_, _, _, A2 = forward_propagation(X)
print("Predicted outputs:\n", A2)


print(W1,'\n', b1,'\n','\n', W2,'\n','\n', b2)