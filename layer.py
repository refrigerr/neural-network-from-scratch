import numpy as np
from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, neuron_input_size):
        self.neurons = [Neuron(neuron_input_size) for _ in range(num_neurons)]

    def feedforward(self, inputs):
        outputs = [neuron.feedforward(inputs) for neuron in self.neurons]
        return np.array(outputs)
    
    def backpropagate(self, output_gradients, learning_rate):

        input_gradients = np.zeros(self.neurons[0].inputs.shape)

        for i, neuron in enumerate(self.neurons):
            
            neuron_output_gradient = output_gradients[i]
            
            neuron_input_gradient = neuron.backpropagate(neuron_output_gradient, learning_rate)
            
            input_gradients += neuron_input_gradient

        
        return input_gradients
