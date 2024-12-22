import numpy as np
from Models.BaseLayer import Layer
class MaxPoolLayer(Layer):
    def __init__(self,kernel_size,stride):
        self.kernel_size = kernel_size
        self.stride = stride
        self.input = None
    def forward(self,X):
        self.input = X
        N,C,H,W = X.shape
        H_f,W_f = self.kernel_size,self.kernel_size
        H_out = (H-H_f) // self.stride + 1
        W_out = (W-W_f) // self.stride + 1

        output = np.zeros((N,C,H_out,W_out))

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = X[n,c,i*self.stride:i*self.stride+H_f,j*self.stride:j*self.stride+W_f]
                        output[n,c,i,j] = np.max(patch)
        return output
    def backward(self,loss,learning_rate):
        N,C,H,W = self.input.shape
        H_f,W_f = self.kernel_size,self.kernel_size
        H_out = (H-H_f) // self.stride + 1
        W_out = (W-W_f) // self.stride + 1
        loss = loss.reshape(N,C,H_out,W_out)

        dx = np.zeros_like(self.input)

        for n in range(N):
            for c in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        patch = self.input[n,c,i*self.stride:i*self.stride+H_f,j*self.stride:j*self.stride+W_f]
                        max_index = np.unravel_index(np.argmax(patch),patch.shape)
                        dx[n,c,i*self.stride + max_index[0],j * self.stride + max_index[1]] += loss[n,c,i,j]
        return dx