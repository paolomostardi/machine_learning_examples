import numpy as np


def sigmoid():
    pass

class neuron :
    def __init__(self, num_inputs ) -> None:
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
        self.activation_func = sigmoid
            
    def bad_predict(self,x):
        sum  = 0
        for index, _ in enumerate(x):
            sum += x[index] * self.weights[index]
        sum += self.bias
        return self.activation_func(sum)
    
    def predict(self, x):
        weighted_sum = np.dot(x, self.weights) + self.bias
        return self.activation_func(weighted_sum)




class mutli_layer: 
    def __init__(self,input_shape, hidden_layers, output = 1) -> None:
        self.input_shape = input_shape
        self.hidden_layers =  []
        
        for index,layer in enumerate(hidden_layers):
            hidden_layer = []

            if index == 0:
                for j in range(layer):
                    hidden_layer.append(neuron(input_shape))
            else:
                for j in range(layer):
                    hidden_layer.append(neuron((hidden_layers[index-1])))

            self.hidden_layers.append(hidden_layer)

    def predicit(self,x):
        if len(x) != self.input_shape:
            print('incorrect shape of input')
            return 
        
        for layer in self.hidden_layers:
            next_x = []
            for neuron in layer:
                next_x.append(neuron(x))
            x = next_x
        
        return self.output(x)




    def print(self):
        for layer in self.hidden_layers:
            print( len (layer))
            for neuron in layer:
                print("   ", neuron.weights)

perc = mutli_layer(2,[3,2,5,1],1)
