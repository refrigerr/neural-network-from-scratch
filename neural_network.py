from layer import Layer
import matplotlib.pyplot as plt

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
    
    def draw_neural_network(self):
        v_spacing = 3
        h_spacing = 4
        radius = 1
        fig, ax = plt.subplots(figsize=(8, 6))

        # Determine vertical positions
        n_layers = len(self.layers_config)
        for i, layer_size in enumerate(self.layers_config):
            x = i * h_spacing
            y_top = (layer_size - 1) * v_spacing / 2
            for j in range(layer_size):
                y = y_top - j * v_spacing
                circle = plt.Circle((x, y), radius=radius, fill=True, color='skyblue', ec='black')
                ax.add_artist(circle)
                ax.text(x, y, f'L{i}N{j}', ha='center', va='center', fontsize=8)

                # Draw connections
                if i > 0:
                    for k in range(self.layers_config[i - 1]):
                        y_prev = ((self.layers_config[i - 1] - 1) * v_spacing / 2) - k * v_spacing
                        ax.plot([x - h_spacing, x], [y_prev, y], 'k-', lw=0.5)

        ax.set_xlim(-1, h_spacing * n_layers)
        ax.set_ylim(-v_spacing * max(self.layers_config), v_spacing * max(self.layers_config))
        ax.axis('off')
        plt.show()

        