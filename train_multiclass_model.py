import numpy as np
from data import Data
from Models.Multiclass_NN import Multiclass_NN

train_images = np.load("MNIST/train_images.npy")
train_labels = np.load("MNIST/train_labels.npy")
test_images = np.load("MNIST/test_images.npy")
test_labels = np.load("MNIST/test_labels.npy")

data = Data(train_images, train_labels, test_images, test_labels)

train = data.train.reshape(-1,784)
train_labels = np.eye(10)[data.train_labels]
test = data.test.reshape(-1,784)
test_labels = np.eye(10)[data.test_labels]

model = Multiclass_NN(784, 128, 10)
model.train(train,train_labels,10)
print("Test: ", model.evaluate(test,test_labels))