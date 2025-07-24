from layer import Layer
import loss_functions

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
    
    def backpropagate(self, y_true, y_pred, learning_rate):
        
        output_gradients = loss_functions.mse_loss_derivative(y_true, y_pred)
        
        for layer in reversed(self.layers):
            output_gradients = layer.backpropagate(output_gradients, learning_rate)
    

    def train_step(self, X, y_true, learning_rate):
        
        y_pred = self.feedforward(X)
        
        loss = loss_functions.mse_loss(y_true, y_pred)
        
        self.backpropagate(y_true, y_pred, learning_rate)
        
        return loss
    
    def train(self, X_train, y_train, epochs, learning_rate, verbose=True):
    
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(len(X_train)):
                loss = self.train_step(X_train[i], y_train[i], learning_rate)
                total_loss += loss
            
            avg_loss = total_loss / len(X_train)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.6f}")


    def predict(self, X):
    
        return self.feedforward(X)