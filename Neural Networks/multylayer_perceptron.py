import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__(self, num_inputs):
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.activation_func = sigmoid
            
    def bad_predict(self, x):
        total_sum = 0
        for index, _ in enumerate(x):
            total_sum += x[index] * self.weights[index]
        total_sum += self.bias
        return self.activation_func(total_sum)
    
    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.activation_func(weighted_sum)

class MultiLayerPerceptron:
    def __init__(self, input_shape, hidden_layers, output=1):
        self.input_shape = input_shape
        self.hidden_layers = []
        self.output_layer = []
        
        # Initialize hidden layers
        for index, layer_size in enumerate(hidden_layers):
            hidden_layer = []
            if index == 0:
                for _ in range(layer_size):
                    hidden_layer.append(Neuron(input_shape))
            else:
                for _ in range(layer_size):
                    hidden_layer.append(Neuron(hidden_layers[index - 1]))
            self.hidden_layers.append(hidden_layer)
        
        # Initialize output layer
        for _ in range(output):
            self.output_layer.append(Neuron(hidden_layers[-1]))
    
    def softmax(self, z):
        exp_z = np.exp(z - np.max(z))  # Numerical stability
        return exp_z / np.sum(exp_z)

    def predict(self, x):
        if len(x) != self.input_shape:
            print('Incorrect shape of input')
            return 
        
        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            next_x = []
            for neuron in layer:
                next_x.append(neuron.predict(x))
            x = next_x
        
        # Forward pass through output layer
        final_outputs = []
        for neuron in self.output_layer:
            final_outputs.append(neuron.predict(x))
        
        # Apply softmax to the output layer's outputs
        return self.softmax(final_outputs)


    def print(self):
        for layer in self.hidden_layers:
            print( len (layer))
            for neuron in layer:
                print("   ", neuron.weights)


a = MultiLayerPerceptron(2,[2,2,2],2)

print(a.predict([1,2]))
