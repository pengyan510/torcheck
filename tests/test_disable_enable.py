import pytest
import torcheck


def test_disable(
    unchanging_model_optimizer,
    unchanging_model,
    dataloader,
    run_training
):
    torcheck.register(unchanging_model_optimizer)
    torcheck.add_module_changing_check(unchanging_model, module_name="NeuralNet")
    torcheck.disable()
    run_training(unchanging_model, dataloader, unchanging_model_optimizer)


def test_disable_enable(
    unchanging_model_optimizer,
    unchanging_model,
    dataloader,
    run_training
):
    torcheck.register(unchanging_model_optimizer)
    torcheck.add_module_changing_check(unchanging_model, module_name="NeuralNet")
    torcheck.disable()
    run_training(unchanging_model, dataloader, unchanging_model_optimizer)
    torcheck.enable()
    with pytest.raises(
        RuntimeError,
        match=r"Module NeuralNet's fc1\.weight should change\.\n.*fc1.bias should change"
    ):
        run_training(unchanging_model, dataloader, unchanging_model_optimizer)
