import numpy as np
class Multiclass_NN:
    def __init__(self,input_size,hidden_size,output_size,learning_rate=0.01,activation_type='relu',batch_size=64,weight_initializer_type='kaiming',regularization='L2',lambda_reg=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.weight_initializer_type = weight_initializer_type
        self.learning_rate = learning_rate
        self.activation_type = activation_type
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        
        #activation function
        if self.activation_type == 'relu':
            self.activation_function = self.relu
            self.activation_derivative = lambda x: np.where(x > 0, 1,0)
        elif self.activation_type == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_derivative = lambda x: x * (1 - x)
        elif self.activation_function == 'tanh':
            self.activation_function = self.tanh
            self.activation_derivative = lambda x: 1 - np.tanh(x) ** 2
        else:
            raise ValueError("Invalid activation function.")
        
        #weight type 
        if self.weight_initializer_type == 'gaussian':
            self.weight_initializer = lambda size: np.random.randn(*size)
        elif self.weight_initializer_type == 'uniform':
            self.weight_initializer = lambda size: np.random.uniform(-1,1, size)
        elif self.weight_initializer_type == 'xavier':
            self.weight_initializer = lambda size: np.random.randn(*size) * np.sqrt(1/size[1])
        elif self.weight_initializer_type == 'kaiming':
            self.weight_initializer = lambda size: np.random.randn(*size) * np.sqrt(2/size[1])
        else:
            raise ValueError("Invalid weight initalizer.")

        #weight initialization
        self.v = self.weight_initializer((self.input_size,self.hidden_size))
        self.b1 = self.weight_initializer((1,self.hidden_size))
        self.w = self.weight_initializer((self.hidden_size,self.output_size))
        self.b2 = self.weight_initializer((1,self.output_size))
    def train(self,x,y,epochs=10):
        for epoch in range(epochs):
            num_batches = int(np.ceil(len(x) / self.batch_size))
            indices = np.arange(len(x))
            np.random.shuffle(indices)
            for num in range(num_batches):
                start = num * self.batch_size
                end = (num + 1) * self.batch_size
                x_batch = x[indices[start:end]]
                y_batch = y[indices[start:end]]
                self.train_batch(x_batch,y_batch)
            print(f"Train Loss for Epoch{epoch}:", self.loss(x,y))
        print("Train Accuracy", self.evaluate(x,y))
    def train_batch(self,x,y):
        #forward pass
        y_pred, h, z = self.forward(x)
        #gradients
        dy = (y_pred - y) / len(x)
        dv = (((dy @ self.w.T) * self.activation_derivative(z)).T @ x).T
        dw = (dy.T @ h).T
        db1 = np.sum(dy @ self.w.T * self.activation_derivative(z) ,axis=0,keepdims=True)
        db2 = np.sum(dy,axis=0, keepdims=True)
        #regularization
        reg_v, reg_w = self.set_regularization()
        #update weights
        self.v -= self.learning_rate * (dv + reg_v)
        self.w -= self.learning_rate * (dw + reg_w)
        self.b1 -= self.learning_rate * db1
        self.b2 -= self.learning_rate * db2
        
    def forward(self,x):
        z = x @ self.v + self.b1
        h = self.activation_function(z)
        y_l = h @ self.w + self.b2
        y_pred = self.softmax(y_l)
        return y_pred, h, z
    def predict(self,x):
        z = x @ self.v + self.b1
        h = self.activation_function(z)
        y_l = h @ self.w + self.b2
        y_pred = self.softmax(y_l)
        return y_pred
    def evaluate(self,x,y):
        y_pred = np.argmax(self.predict(x),axis=1)
        y = np.argmax(y,axis=1)
        accuracy = np.mean(y_pred == y)
        return accuracy
    def loss(self,x,y):
        y_pred = self.predict(x)
        loss = -np.mean(np.sum(y * np.log(y_pred + 1e-8),axis=1))
        return loss
    def relu(self,x):
        return np.maximum(0,x)
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def tanh(self,x):
        return np.tanh(x)
    def softmax(self,z):
        z_exp = np.exp(z - np.max(z))
        return z_exp / (z_exp.sum(axis=1, keepdims=True) + 1e-8)
    def set_regularization(self):
        if self.regularization == 'L1':
            return self.lambda_reg * np.sign(self.v), self.lambda_reg * np.sign(self.w)
        if self.regularization == 'L2':
            return self.lambda_reg * self.v, self.lambda_reg * self.w
        else:
            return 0,0
    
