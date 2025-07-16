import numpy as np
from neuron import Neuron

class Layer:
    def __init__(self, num_neurons, neuron_input_size):
        self.neurons = [Neuron(neuron_input_size) for _ in range(num_neurons)]

    def feedforward(self, inputs):
        outputs = [neuron.feedforward(inputs) for neuron in self.neurons]
        return np.array(outputs)