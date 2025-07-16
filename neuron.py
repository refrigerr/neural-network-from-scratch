import numpy as np
import activation_functions

class Neuron:
    def __init__(self, num_imputs):
        self.weights = np.random.randn(num_imputs)
        self.bias = np.random.randn()

    def feedforward(self, inputs):
        self.inputs = inputs
        self.total = np.dot(self.weights, inputs) + self.bias #input1 * weight1 + input2 * weight2 + input3 * weight3 + ... + bias
        self.output = activation_functions.sigmoid(self.total)
        return self.output
    

