import pytest
import torcheck


def test_verbose_on(
    unchanging_model_optimizer, unchanging_model, dataloader, run_training
):
    torcheck.verbose_on()
    torcheck.register(unchanging_model_optimizer)
    torcheck.add_module_changing_check(unchanging_model, module_name="NeuralNet")
    with pytest.raises(
        RuntimeError,
        match=(
            r"Module NeuralNet's fc1\.weight should change\.\n"
            r"The tensor is:(.|\n)*"
            r"fc1\.bias should change\.\n"
            r"The tensor is:(.|\n)*"
        ),
    ):
        run_training(unchanging_model, dataloader, unchanging_model_optimizer)


def test_verbose_off(
    unchanging_model_optimizer, unchanging_model, dataloader, run_training
):
    torcheck.register(unchanging_model_optimizer)
    torcheck.add_module_changing_check(unchanging_model, module_name="NeuralNet")
    torcheck.verbose_on()
    with pytest.raises(RuntimeError):
        run_training(unchanging_model, dataloader, unchanging_model_optimizer)
    torcheck.verbose_off()
    with pytest.raises(
        RuntimeError,
        match=(
            r"Module NeuralNet's fc1\.weight should change\.\n"
            r".*fc1.bias should change"
        ),
    ):
        run_training(unchanging_model, dataloader, unchanging_model_optimizer)
