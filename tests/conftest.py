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


class NaNNet(NetBase):

    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.fc2(output)
        output = output - torch.max(output)
        output = torch.sqrt(output)
        return output

        
class InfNet(NetBase):

    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.fc2(output)
        output = output / 0
        return output

        
class BoundedNet(NetBase):

    def forward(self, x):
        output = self.relu(self.fc1(x))
        output = self.fc2(output)
        output = F.softmax(output, dim=1)
        return output

        
@pytest.fixture(scope="function")
def changing_model():
    return ChangingNet()


@pytest.fixture(scope="function")
def unchanging_model():
    return UnchangingNet()


@pytest.fixture(scope="function")
def nan_model():
    return NaNNet()


@pytest.fixture(scope="function")
def inf_model():
    return InfNet()


@pytest.fixture(scope="function")
def bounded_model():
    return BoundedNet()


@pytest.fixture(scope="function")
def changing_model_optimizer(changing_model):
    return Adam(changing_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def unchanging_model_optimizer(unchanging_model):
    return Adam(unchanging_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def nan_model_optimizer(nan_model):
    return Adam(nan_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def inf_model_optimizer(inf_model):
    return Adam(inf_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def bounded_model_optimizer(bounded_model):
    return Adam(bounded_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def correct_model_optimizer(correct_model):
    return Adam(correct_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def nonan_model_optimizer(nonan_model):
    return Adam(nonan_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def noinf_model_optimizer(noinf_model):
    return Adam(noinf_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def unbounded_model_optimizer(unbounded_model):
    return Adam(unbounded_model.parameters(), lr=0.001)


@pytest.fixture(scope="function")
def dataloader():
    torch.manual_seed(42)
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


correct_model = changing_model
nonan_model = changing_model
noinf_model = changing_model
unbounded_model = changing_model

