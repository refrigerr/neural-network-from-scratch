from layer import Layer

class NeuralNetwork:
    def __init__(self, layers_config):
    
        self.layers_config = layers_config
        self.layers = []
        for i in range(1, len(layers_config)): #first layer ommited as it is input
            layer = Layer(num_neurons=layers_config[i], neuron_input_size=layers_config[i - 1])
            self.layers.append(layer)

    def feedforward(self, inputs):
        for layer in self.layers:
            inputs = layer.feedforward(inputs)
        return inputs
    