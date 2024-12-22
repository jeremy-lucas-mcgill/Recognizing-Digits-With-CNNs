import torch
import numpy as np
from torch.utils.data import Dataset

class NumpyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data[:, np.newaxis, :, :]
        self.data = self.data / 255
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
