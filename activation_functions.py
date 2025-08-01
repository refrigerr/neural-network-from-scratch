import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
        sx = sigmoid(x)
        return sx * (1 - sx)