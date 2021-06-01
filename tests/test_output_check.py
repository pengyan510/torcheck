import pytest
import torcheck


def test_module_output_range_check_with_bounded_model(
    bounded_model_optimizer, bounded_model, dataloader, run_training
):
    torcheck.add_module_output_range_check(
        bounded_model, output_range=(0, 1), module_name="NeuralNet"
    )
    run_training(bounded_model, dataloader, bounded_model_optimizer)


def test_module_output_range_check_with_unbounded_model(
    unbounded_model_optimizer, unbounded_model, dataloader, run_training
):
    torcheck.add_module_output_range_check(
        unbounded_model, output_range=(0, 1), module_name="NeuralNet"
    )
    with pytest.raises(
        RuntimeError, match=r"Module NeuralNet's output should all > 0 and < 1"
    ):
        run_training(unbounded_model, dataloader, unbounded_model_optimizer)


def test_module_output_negate_range_check_with_bounded_model(
    bounded_model_optimizer, bounded_model, dataloader, run_training
):
    torcheck.add_module_output_range_check(
        bounded_model,
        output_range=(0, 1),
        negate_range=True,
        module_name="NeuralNet",
    )
    with pytest.raises(
        RuntimeError,
        match=r"Module NeuralNet's output shouldn't all > 0 and < 1",
    ):
        run_training(bounded_model, dataloader, bounded_model_optimizer)


def test_module_output_negate_range_check_with_unbounded_model(
    unbounded_model_optimizer, unbounded_model, dataloader, run_training
):
    torcheck.add_module_output_range_check(
        unbounded_model,
        output_range=(0, 1),
        negate_range=True,
        module_name="NeuralNet",
    )
    run_training(unbounded_model, dataloader, unbounded_model_optimizer)
