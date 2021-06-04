import pytest
import torcheck


def test_module_changing_check_with_changing_model(
    changing_model_optimizer, changing_model, dataloader, run_training
):
    torcheck.register(changing_model_optimizer)
    torcheck.add_module_changing_check(changing_model, module_name="NeuralNet")
    run_training(changing_model, dataloader, changing_model_optimizer)


def test_module_changing_check_with_unchanging_model(
    unchanging_model_optimizer, unchanging_model, dataloader, run_training
):
    torcheck.register(unchanging_model_optimizer)
    torcheck.add_module_changing_check(unchanging_model, module_name="NeuralNet")
    with pytest.raises(
        RuntimeError,
        match=(
            r"Module NeuralNet's fc1\.weight should change\.\n"
            r".*fc1.bias should change"
        ),
    ):
        run_training(unchanging_model, dataloader, unchanging_model_optimizer)


def test_module_unchanging_check_with_changing_model(
    changing_model_optimizer, changing_model, dataloader, run_training
):
    torcheck.register(changing_model_optimizer)
    torcheck.add_module_unchanging_check(changing_model, module_name="NeuralNet")
    with pytest.raises(
        RuntimeError,
        match=(
            r"Module NeuralNet's fc1\.weight should not change\."
            r"(.|\n)*fc2\.weight should not change"
        ),
    ):
        run_training(changing_model, dataloader, changing_model_optimizer)


def test_module_unchanging_check_with_unchanging_model(
    unchanging_model_optimizer, unchanging_model, dataloader, run_training
):
    torcheck.register(unchanging_model_optimizer)
    torcheck.add_module_unchanging_check(
        unchanging_model.fc1, module_name="First Layer"
    )
    run_training(unchanging_model, dataloader, unchanging_model_optimizer)


def test_tensor_changing_check_with_changing_model(
    changing_model_optimizer, changing_model, dataloader, run_training
):
    torcheck.register(changing_model_optimizer)
    torcheck.add_tensor_changing_check(
        changing_model.fc1.weight, tensor_name="fc1.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_changing_check(
        changing_model.fc1.bias, tensor_name="fc1.bias", module_name="NeuralNet"
    )
    torcheck.add_tensor_changing_check(
        changing_model.fc2.weight, tensor_name="fc2.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_changing_check(
        changing_model.fc2.bias, tensor_name="fc2.bias", module_name="NeuralNet"
    )
    run_training(changing_model, dataloader, changing_model_optimizer)


def test_tensor_changing_check_with_unchanging_model(
    unchanging_model_optimizer, unchanging_model, dataloader, run_training
):
    torcheck.register(unchanging_model_optimizer)
    torcheck.add_tensor_changing_check(
        unchanging_model.fc1.weight, tensor_name="fc1.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_changing_check(
        unchanging_model.fc1.bias, tensor_name="fc1.bias", module_name="NeuralNet"
    )
    torcheck.add_tensor_changing_check(
        unchanging_model.fc2.weight, tensor_name="fc2.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_changing_check(
        unchanging_model.fc2.bias, tensor_name="fc2.bias", module_name="NeuralNet"
    )
    with pytest.raises(
        RuntimeError,
        match=(
            r"Module NeuralNet's fc1\.weight should change\.\n"
            r".*fc1.bias should change"
        ),
    ):
        run_training(unchanging_model, dataloader, unchanging_model_optimizer)


def test_tensor_unchanging_check_with_changing_model(
    changing_model_optimizer, changing_model, dataloader, run_training
):
    torcheck.register(changing_model_optimizer)
    torcheck.add_tensor_unchanging_check(
        changing_model.fc1.weight, tensor_name="fc1.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_unchanging_check(
        changing_model.fc1.bias, tensor_name="fc1.bias", module_name="NeuralNet"
    )
    with pytest.raises(
        RuntimeError,
        match=(
            r"Module NeuralNet's fc1\.weight should not change\.\n"
            r".*fc1.bias should not change"
        ),
    ):
        run_training(changing_model, dataloader, changing_model_optimizer)


def test_tensor_unchanging_check_with_unchanging_model(
    unchanging_model_optimizer, unchanging_model, dataloader, run_training
):
    torcheck.register(unchanging_model_optimizer)
    torcheck.add_tensor_unchanging_check(
        unchanging_model.fc1.weight, tensor_name="fc1.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_unchanging_check(
        unchanging_model.fc1.bias, tensor_name="fc1.bias", module_name="NeuralNet"
    )
    run_training(unchanging_model, dataloader, unchanging_model_optimizer)


def test_tensor_nan_check_with_nan_model(
    nan_model_optimizer, nan_model, dataloader, run_training
):
    torcheck.register(nan_model_optimizer)
    torcheck.add_tensor_nan_check(
        nan_model.fc1.weight, tensor_name="fc1.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_nan_check(
        nan_model.fc1.bias, tensor_name="fc1.bias", module_name="NeuralNet"
    )
    torcheck.add_tensor_nan_check(
        nan_model.fc2.weight, tensor_name="fc2.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_nan_check(
        nan_model.fc2.bias, tensor_name="fc2.bias", module_name="NeuralNet"
    )
    with pytest.raises(
        RuntimeError,
        match=(
            r"Module NeuralNet's fc1\.weight contains NaN\.\n"
            r".*fc1.bias contains NaN\.\n.*fc2.weight contains NaN\.\n"
            r".*fc2.bias contains NaN"
        ),
    ):
        run_training(nan_model, dataloader, nan_model_optimizer)


def test_tensor_nan_check_with_nonan_model(
    nonan_model_optimizer, nonan_model, dataloader, run_training
):
    torcheck.register(nonan_model_optimizer)
    torcheck.add_tensor_nan_check(
        nonan_model.fc1.weight, tensor_name="fc1.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_nan_check(
        nonan_model.fc1.bias, tensor_name="fc1.bias", module_name="NeuralNet"
    )
    torcheck.add_tensor_nan_check(
        nonan_model.fc2.weight, tensor_name="fc2.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_nan_check(
        nonan_model.fc2.bias, tensor_name="fc2.bias", module_name="NeuralNet"
    )
    run_training(nonan_model, dataloader, nonan_model_optimizer)


def _test_tensor_inf_check_with_inf_model(
    inf_model_optimizer, inf_model, dataloader, run_training
):
    """TODO: design a test case with inf gradient values"""
    torcheck.register(inf_model_optimizer)
    torcheck.add_tensor_inf_check(
        inf_model.fc1.weight, tensor_name="fc1.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_inf_check(
        inf_model.fc1.bias, tensor_name="fc1.bias", module_name="NeuralNet"
    )
    torcheck.add_tensor_inf_check(
        inf_model.fc2.weight, tensor_name="fc2.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_inf_check(
        inf_model.fc2.bias, tensor_name="fc2.bias", module_name="NeuralNet"
    )
    with pytest.raises(
        RuntimeError,
        match=(
            r"Module NeuralNet's fc1\.weight contains inf\.\n"
            r".*fc1.bias contains inf\.\n.*fc2.weight contains inf\.\n"
            r".*fc2.bias contains inf"
        ),
    ):
        run_training(inf_model, dataloader, inf_model_optimizer)


def test_tensor_inf_check_with_noinf_model(
    noinf_model_optimizer, noinf_model, dataloader, run_training
):
    torcheck.register(noinf_model_optimizer)
    torcheck.add_tensor_inf_check(
        noinf_model.fc1.weight, tensor_name="fc1.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_inf_check(
        noinf_model.fc1.bias, tensor_name="fc1.bias", module_name="NeuralNet"
    )
    torcheck.add_tensor_inf_check(
        noinf_model.fc2.weight, tensor_name="fc2.weight", module_name="NeuralNet"
    )
    torcheck.add_tensor_inf_check(
        noinf_model.fc2.bias, tensor_name="fc2.bias", module_name="NeuralNet"
    )
    run_training(noinf_model, dataloader, noinf_model_optimizer)


def test_tensor_multiple_check_with_correct_model(
    correct_model_optimizer, correct_model, dataloader, run_training
):
    torcheck.register(correct_model_optimizer)
    torcheck.add_tensor(
        correct_model.fc1.weight,
        tensor_name="fc1.weight",
        module_name="NeuralNet",
        changing=True,
        check_nan=True,
        check_inf=True,
    )
    torcheck.add_tensor(
        correct_model.fc1.bias,
        tensor_name="fc1.bias",
        module_name="NeuralNet",
        changing=True,
        check_nan=True,
        check_inf=True,
    )
    run_training(correct_model, dataloader, correct_model_optimizer)
