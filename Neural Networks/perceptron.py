
import matplotlib.pyplot as plt

# bonus points for less usefull feuature and normalization 

def heaviside_step_function(x):
    if x > 0:
        return 1
    return 0

def perceptron( w, x, b, O = heaviside_step_function ):
        
    sum = 0
    for i in range(len(w)):
        sum += w[i] * x[i]
    sum += b
    return O(sum)

def training(learning_rate, training_set, weights, bias):
    r = learning_rate
    w = weights
    b = bias

    total_error = 0
    
    for input_vector, target in training_set:
        y_j = perceptron(w, input_vector, b)
        error = target - y_j
        total_error += abs(error)
        
        for i, w_i in enumerate(w):
            w[i] = w_i + r * error * input_vector[i]
        
        b = b + r * error

    average_error = total_error / len(training_set)
    return w, b, average_error


training_set = [[[0,0],0],
                [[0,1],0],
                [[-1,0],0],
                [[0,-1],0],
                [[1,1],1],
                [[2,1],1], 
                [[3,1],1],
                [[4,1],1], 
                [[0.2,0.5],0],      
                [[0.2,0.5],0],
                [[0.8,0.5],0],
                [[0.8,0.9],0],
                [[1.8,0.9],1], 
                [[1.8,0.4],0],        
                [[1,0.4],0],    
                [[2,0.4],0], 
                [[3,0.4],0],
                [[3,0.1],0],
                [[3,0],0], 
                [[3,0.5],1], 
                [[2.9,0.5],1], 
                ]


weights = [0.5,0.5]
bias = 0
average_error = 0

for i in range(10):
    print(weights,bias,average_error)
    weights, bias, average_error = training(0.1,training_set,weights,bias)


# Separate the features and labels
features = [item[0] for item in training_set]
labels = [item[1] for item in training_set]

# Create lists for the x and y coordinates
x_coords = [feature[0] for feature in features]
y_coords = [feature[1] for feature in features]

# Create the scatter plot
plt.figure(figsize=(8, 6))
for i in range(len(labels)):
    if labels[i] == 0:
        plt.scatter(x_coords[i], y_coords[i], color='red', marker='o', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(x_coords[i], y_coords[i], color='blue', marker='x', label='Class 1' if i == 4 else "")

# Adding labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Training Set')
plt.legend()
plt.grid(True)
plt.show()
