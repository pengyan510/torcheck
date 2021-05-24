import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class NetBase(nn.Module):
    
    def __init__(self):
        super(NetBase, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()


class CorrectNet(NetBase):

    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.fc2(output)
        return output


class NotChangingNet(NetBase):

    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.fc2(x)
        return output


@pytest.fixture
def correct_model():
    return CorrectNet()


@pytest.fixture
def notchanging_model():
    return NotChangingNet()


@pytest.fixture
def dataloader():
    x_data = torch.randn(8, 5)
    y_data = torch.randint(low=0, high=2, size=(8,))
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=4)

