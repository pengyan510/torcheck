import pytest
import torcheck


def test_module_changing_check_with_changing_model(
    changing_model_optimizer,
    changing_model,
    dataloader,
    run_training
):
    torcheck.register(changing_model_optimizer)
    torcheck.add_module_changing_check(changing_model, module_name="NeuralNet")
    run_training(changing_model, dataloader, changing_model_optimizer)


def test_module_changing_check_with_unchanging_model(
    unchanging_model_optimizer,
    unchanging_model,
    dataloader,
    run_training
):
    torcheck.register(unchanging_model_optimizer)
    torcheck.add_module_changing_check(unchanging_model, module_name="NeuralNet")
    with pytest.raises(
        RuntimeError,
        match=r"Module NeuralNet's fc1\.weight should change\.\n.*fc1.bias should change"
    ):
        run_training(unchanging_model, dataloader, unchanging_model_optimizer)


def test_module_unchanging_check_with_changing_model(
    changing_model_optimizer,
    changing_model,
    dataloader,
    run_training
):
    torcheck.register(changing_model_optimizer)
    torcheck.add_module_unchanging_check(changing_model, module_name="NeuralNet")
    with pytest.raises(
        RuntimeError,
        match=r"Module NeuralNet's fc1\.weight should not change\.(.|\n)*fc2\.weight should not change"
    ):
        run_training(changing_model, dataloader, changing_model_optimizer)


def test_module_unchanging_check_with_unchanging_model(
    unchanging_model_optimizer,
    unchanging_model,
    dataloader,
    run_training
):
    torcheck.register(unchanging_model_optimizer)
    torcheck.add_module_unchanging_check(unchanging_model.fc1, module_name="First Layer")
    run_training(unchanging_model, dataloader, unchanging_model_optimizer)

