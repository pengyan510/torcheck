import pytest
import torch
import torch.nn as nn


class NetBase(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(5, 3),
        self.fc2 = nn.Linear(3, 2),
        self.relu = nn.ReLu()


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


@pytest.fixture(scope="function")
def correct_model():
    return CorrectNet()


@pytest.fixture(scope="function")
def not_changing_model():
    return NotChangingNet()


@pytest.fixture(scope="function")
def dataloader():
    
