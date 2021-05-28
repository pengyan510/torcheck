import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader


class NetBase(nn.Module):
    
    def __init__(self):
        super(NetBase, self).__init__()
        self.fc1 = nn.Linear(5, 5)
        self.fc2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()


class ChangingNet(NetBase):

    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.fc2(output)
        return output


class UnchangingNet(NetBase):

    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.fc2(x)
        return output


@pytest.fixture(scope="function")
def changing_model():
    return ChangingNet()


@pytest.fixture(scope="function")
def unchanging_model():
    return UnchangingNet()


@pytest.fixture(scope="function")
def changing_model_optimizer(changing_model):
    return Adam(changing_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def unchanging_model_optimizer(unchanging_model):
    return Adam(unchanging_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def dataloader():
    x_data = torch.randn(8, 5)
    y_data = torch.randint(low=0, high=2, size=(8,))
    dataset = TensorDataset(x_data, y_data)
    return DataLoader(dataset, batch_size=4)


@pytest.fixture(scope="function")
def run_training():

    def func(model, dataloader, optimizer):
        for x_from_data, y_from_data in dataloader:
            y_from_model = model(x_from_data)
            loss = F.cross_entropy(y_from_model, y_from_data)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    return func

