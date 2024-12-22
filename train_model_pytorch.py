from data_pytorch import NumpyDataset
from cnn_pytorch import pytorchCNN
from torch.utils.data import DataLoader
import torch
import numpy as np
import os

train_data = np.load("MNIST/train_images.npy")
train_labels = np.load("MNIST/train_labels.npy")
test_data = np.load("MNIST/test_images.npy")
test_labels = np.load("MNIST/test_labels.npy")

train_dataset = NumpyDataset(train_data, train_labels)
test_dataset = NumpyDataset(test_data, test_labels)

trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model = pytorchCNN()
model_name = "pytorchCNN.pth"

if (os.path.exists(model_name)):
    model.load_state_dict(torch.load(model_name))
    model.eval()
else:
    model.train_model(trainloader, num_epochs=10, learning_rate=0.001)
    torch.save(model.state_dict(), model_name)

accuracy = model.evaluate(testloader)

