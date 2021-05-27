import pytest
import torch
import torch.nn.functional as F
from torch.optim import Adam
import torcheck


def run_training(model, dataloader, optimizer):
    for x_from_data, y_from_data in dataloader:
        y_from_model = model(x_from_data)
        loss = F.cross_entropy(y_from_model, y_from_data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_changing_module_with_correct_model(correct_model, dataloader):
    optimizer = Adam(correct_model.parameters(), lr=0.001)
    torcheck.register(optimizer)
    torcheck.add_module_changing_check(correct_model, module_name="NeuralNet")
    run_training(correct_model, dataloader, optimizer)


def test_changing_module_with_notchanging_model(notchanging_model, dataloader):
    optimizer = Adam(notchanging_model.parameters(), lr=0.001)
    torcheck.register(optimizer)
    torcheck.add_module_changing_check(notchanging_model, module_name="NeuralNet")
    with pytest.raises(
        RuntimeError,
        match=r"Module NeuralNet's fc1\.weight should change\.\n.*fc1.bias should change"
    ):
        run_training(notchanging_model, dataloader, optimizer)

