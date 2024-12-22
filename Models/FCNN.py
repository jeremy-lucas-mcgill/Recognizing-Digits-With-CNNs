import numpy as np
from Models.BaseLayer import Layer
class FCNNLayer(Layer):
    def __init__(self,input_size,hidden_nodes,activation_function,weight_initialization):
        super().__init__(activation_function, weight_initialization)
        self.input_size = input_size
        self.hidden_nodes = hidden_nodes
        self.pre_activation_output = None
        self.input = None
        self.weights = self.weight_initializer((hidden_nodes,input_size))
        self.bias = np.zeros((1,hidden_nodes))
    def forward(self,X):
        X = X.reshape(X.shape[0],-1)
        self.input = X
        self.pre_activation_output = self.input @ self.weights.T + self.bias
        output = self.activation_function(self.pre_activation_output)
        return output
    def backward(self,loss,learning_rate):
        activation_derivative = self.activation_derivative_function(self.pre_activation_output)
        weight_derivative = self.input
        new_loss = (loss * activation_derivative) @ self.weights
        dw = (loss * activation_derivative).T @ weight_derivative
        db = np.sum(loss * activation_derivative,axis=0,keepdims=True)
        self.weights -= learning_rate * dw
        self.bias -= learning_rate * db
        return new_loss
   