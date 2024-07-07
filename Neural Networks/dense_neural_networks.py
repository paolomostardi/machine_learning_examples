import numpy as np


def heaviside_step_function(x):
    if x > 0:
        return 1
    return 0

def perceptron( O = heaviside_step_function, w, x, b ):
        
    sum = 0
    for i in range(len(w)):
        sum += w[i] * x[i]
    sum += b
    return O(sum)

def training():

    learning_rate = 0.1
    
    pass


