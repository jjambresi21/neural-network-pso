import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.total_params = (input_size * hidden_size) + hidden_size + (hidden_size * output_size) + output_size
        
    def initialize_parameters(self):
        W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        b1 = np.zeros((1, self.hidden_size))
        W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        b2 = np.zeros((1, self.output_size))
        return W1, b1, W2, b2

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forwardpropagation(self, X, weights=None, W1=None, b1=None, W2=None, b2=None):
        if weights is not None:
            W1, b1, W2, b2 = self.decode_weights(weights)
            
        hidden_input = np.dot(X, W1) + b1
        hidden_output = self.sigmoid(hidden_input)
        return np.dot(hidden_output, W2) + b2

    @staticmethod
    def calculate_mse_loss(y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def decode_weights(self, weights):
        w1_end = self.input_size * self.hidden_size
        b1_end = w1_end + self.hidden_size
        w2_end = b1_end + self.hidden_size * self.output_size
        b2_end = w2_end + self.output_size

        W1 = weights[:w1_end].reshape((self.input_size, self.hidden_size))
        b1 = weights[w1_end:b1_end].reshape((1, self.hidden_size))
        W2 = weights[b1_end:w2_end].reshape((self.hidden_size, self.output_size))
        b2 = weights[w2_end:b2_end].reshape((1, self.output_size))

        return W1, b1, W2, b2
    
    def evaluate_weights(self, weights, X, y):
        W1, b1, W2, b2 = self.decode_weights(weights)
        y_pred = self.forwardpropagation(X, W1=W1, b1=b1, W2=W2, b2=b2)
        return self.calculate_mse_loss(y, y_pred)

    def predict(self, new_input, weights=None, W1=None, b1=None, W2=None, b2=None):
        if weights is not None:
            W1, b1, W2, b2 = self.decode_weights(weights)
            
        hidden_input = np.dot(new_input, W1) + b1
        hidden_output = self.sigmoid(hidden_input)
        return np.dot(hidden_output, W2) + b2