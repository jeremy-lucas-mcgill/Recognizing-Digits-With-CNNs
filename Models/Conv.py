from Models.BaseLayer import Layer
import numpy as np 
class ConvLayer(Layer):
    def __init__(self,in_channels,out_channels,filter_size,stride,padding,activation_function,weight_initializer):
        super().__init__(activation_function,weight_initializer)
        self.filters = self.weight_initializer((out_channels,in_channels,filter_size,filter_size))
        self.bias = self.weight_initializer((1,out_channels))
        self.padding = padding
        self.stride = stride
        self.input = None

    def forward(self,X):
        self.input = X
        N,C_in,H,W = X.shape
        C_out,C_in_filter,H_f,W_f = self.filters.shape
        H_out = (H + 2 * self.padding - H_f)//self.stride + 1
        W_out = (W + 2 * self.padding - W_f)//self.stride + 1
        output = np.zeros((N,C_out,H_out,W_out))
        X_padded = np.pad(X,((0,0),(0,0),(self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        for n in range(N):
            for co in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = X_padded[n,:,i*self.stride:i*self.stride+H_f,j*self.stride:j*self.stride+W_f]
                        output[n,co,i,j] = np.sum(patch * self.filters[co]) + self.bias[0,co]
        return output
    def backward(self,loss,learning_rate):
        X = self.input
        N,C_in,H,W = X.shape
        C_out,C_in_filter,H_f,W_f = self.filters.shape
        _, _, H_out,W_out = loss.shape

        dx = np.zeros_like(X)
        dw = np.zeros_like(self.filters)
        db = np.zeros_like(self.bias)

        X_padded = np.pad(X,((0,0),(0,0),(self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        dx_padded = np.pad(dx,((0,0),(0,0),(self.padding,self.padding), (self.padding,self.padding)), mode='constant')
        for n in range(N):
            for co in range(C_out):
                    for i in range(H_out):
                        for j in range(W_out):
                            patch = X_padded[n,:,i*self.stride:i*self.stride+H_f,j*self.stride:j*self.stride+W_f]
                            dw[co] += patch * loss[n,co,i,j]
                            db[0,co] += loss[n,co,i,j]
                            dx_padded[n,:,i*self.stride:i*self.stride+H_f,j*self.stride:j*self.stride+W_f] += self.filters[co] * loss[n,co,i,j]
        dx = dx_padded[:,:,self.padding:self.padding+H,self.padding:self.padding+W]
        
        self.filters -= learning_rate * dw
        self.bias -= learning_rate * db
        return dx
        