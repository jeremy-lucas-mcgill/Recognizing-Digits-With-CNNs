from Models.CNNmodel import *
from data import Data

train_images = np.load("MNIST/train_images.npy")
train_labels = np.load("MNIST/train_labels.npy")
test_images = np.load("MNIST/test_images.npy")
test_labels = np.load("MNIST/test_labels.npy")

data = Data(train_images, train_labels, test_images, test_labels)

train = data.train.reshape(-1,1,28,28)
train_labels = np.eye(10)[data.train_labels]
test = data.test.reshape(-1,1,28,28)
test_labels = np.eye(10)[data.test_labels]

model = CNNModel('CEL',None)
model.addConvLayer(1,2,3,1,1,'relu','kaiming')
model.addMaxPoolLayer(2,2)
model.addFCNNLayer(2*14*14,64,'relu','kaiming')
model.addFCNNLayer(64,10,'linear','kaiming')
model.start_train(train[:100],train_labels[:100],1,8,0.01)
test_accuracy = model.evaluate(test[:100],test_labels[:100])
print("Test Accuracy: ",test_accuracy)