import numpy as np
from Models.FCNN import FCNNLayer
from Models.Conv import ConvLayer
from Models.MaxPool import MaxPoolLayer
class CNNModel:
    def __init__(self,loss_function,optimizer):
        self.loss_function,self.backward_loss_function,self.output_function = self.get_loss_function(loss_function)
        self.optimizer = optimizer
        self.layers = []
    def addFCNNLayer(self,input_size,hidden_nodes,activation_function,weight_initialization):
        layer = FCNNLayer(input_size,hidden_nodes,activation_function,weight_initialization)
        self.layers.append(layer)
    def addConvLayer(self,in_channels,out_channels,filter_size,stride,padding,activation_function,weight_initializer):
        layer = ConvLayer(in_channels,out_channels,filter_size,stride,padding,activation_function,weight_initializer)
        self.layers.append(layer)
    def addMaxPoolLayer(self,kernel_size,stride):
        layer = MaxPoolLayer(kernel_size,stride)
        self.layers.append(layer)
    def setLossFunction(self,loss_function):
        self.loss_function = loss_function
    def setOptimizer(self,optimizer):
        self.optimizer = optimizer
    def start_train(self,X,y,epochs,batch_size,learning_rate):
        for epoch in range(epochs):
            num_batches = int(np.ceil(len(X)/batch_size))
            indicies = np.arange(0,len(X))
            np.random.shuffle(indicies)
            for i in range(num_batches):
                start = i * batch_size
                batchX = X[indicies[start: start + batch_size]]
                batchY = y[indicies[start: start + batch_size]]
                self.train_batch(batchX,batchY,learning_rate)
            loss = self.get_loss(X,y)
            print(f"Epoch {epoch} Loss: {loss:.4f}")
        print("Training Accuracy: ", self.evaluate(X,y))
    def train_batch(self,batchX,batchY,learning_rate):
        y_pred = self.predict(batchX)
        backward_loss = self.backward_loss_function(batchY,y_pred)
        for layer in reversed(self.layers):
            backward_loss = layer.backward(backward_loss,learning_rate)
    def get_loss_function(self,loss_function):
        if loss_function == 'CEL':
            loss = lambda y,y_pred: -np.mean(np.sum(y * np.log(y_pred + 1e-8),axis=1))
            backward_loss = lambda y,y_pred: (y_pred-y)/len(y)
            output_function = self.softmax
            return loss,backward_loss,output_function
    def evaluate(self,X,y):
        y_pred = self.predict(X)
        y_pred = np.argmax(y_pred,axis=1)
        y = np.argmax(y,axis=1)
        return np.mean(y_pred == y)
    def get_loss(self,X,y):
        y_pred = self.predict(X)
        return self.loss_function(y,y_pred)
    def predict(self,X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return self.output_function(output)
    def softmax(self,z):
        z_exp = np.exp(z - np.max(z))
        return z_exp / (z_exp.sum(axis=1, keepdims=True) + 1e-8)