import numpy as np
import activation_functions

class Neuron:
    def __init__(self, num_imputs):
        self.weights = np.random.randn(num_imputs)
        self.bias = np.random.randn()

        self.inputs = None
        self.total = None
        self.output = None
        

    def feedforward(self, inputs):
        self.inputs = inputs
        self.total = np.dot(self.weights, inputs) + self.bias #input1 * weight1 + input2 * weight2 + input3 * weight3 + ... + bias
        self.output = activation_functions.sigmoid(self.total)
        return self.output
    
    def backpropagate(self, output_gradient, learning_rate):

        total_gradient = output_gradient * activation_functions.sigmoid_derivative(self.total)

        weight_gradients = total_gradient * self.inputs
        bias_gradient = total_gradient

        input_gradient = total_gradient * self.weights

        self.weights -= learning_rate * weight_gradients
        self.bias -= learning_rate * bias_gradient

        return input_gradient

