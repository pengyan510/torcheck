import pytest
import torcheck


def test_module_nan_check_with_nan_model(
    nan_model_optimizer,
    nan_model,
    dataloader,
    run_training
):
    torcheck.register(nan_model_optimizer)
    torcheck.add_module_nan_check(nan_model, module_name="NeuralNet")
    with pytest.raises(
        RuntimeError,
        match=r"Module NeuralNet's output contains NaN"
    ):
        run_training(nan_model, dataloader, nan_model_optimizer)


def test_module_nan_check_with_nonan_model(
    nonan_model_optimizer,
    nonan_model,
    dataloader,
    run_training
):
    torcheck.register(nonan_model_optimizer)
    torcheck.add_module_nan_check(nonan_model, module_name="NeuralNet")
    run_training(nonan_model, dataloader, nonan_model_optimizer)


def test_module_inf_check_with_inf_model(
    inf_model_optimizer,
    inf_model,
    dataloader,
    run_training
):
    torcheck.register(inf_model_optimizer)
    torcheck.add_module_inf_check(inf_model, module_name="NeuralNet")
    with pytest.raises(
        RuntimeError,
        match=r"Module NeuralNet's output contains inf"
    ):
        run_training(inf_model, dataloader, inf_model_optimizer)


def test_module_inf_check_with_noinf_model(
    noinf_model_optimizer,
    noinf_model,
    dataloader,
    run_training
):
    torcheck.register(noinf_model_optimizer)
    torcheck.add_module_inf_check(noinf_model, module_name="NeuralNet")
    run_training(noinf_model, dataloader, noinf_model_optimizer)


def test_module_multiple_check_with_correct_model(
    correct_model_optimizer,
    correct_model,
    dataloader,
    run_training
):
    torcheck.register(correct_model_optimizer)
    torcheck.add_module(
        correct_model,
        module_name="NeuralNet",
        changing=True,
        output_range=(0, 1),
        negate_range=True,
        check_nan=True,
        check_inf=True
    )
    run_training(correct_model, dataloader, correct_model_optimizer)

