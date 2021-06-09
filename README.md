# torcheck
[![Build Status](https://travis-ci.com/pengyan510/torcheck.svg?branch=master)](https://travis-ci.com/pengyan510/torcheck)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/pengyan510/torcheck/branch/master/graph/badge.svg?token=Q8ADT16N8A)](https://codecov.io/gh/pengyan510/torcheck)
[![PyPI version](https://badge.fury.io/py/torcheck.svg)](https://badge.fury.io/py/torcheck)

Torcheck is a machine learning sanity check toolkit for PyTorch.

For a general introduction, please check this out: [Testing Your PyTorch Models with Torcheck](https://towardsdatascience.com/testing-your-pytorch-models-with-torcheck-cb689ecbc08c)

## About
The creation of torcheck is inspired by Chase Roberts' [Medium post](https://thenerdstation.medium.com/mltest-automatically-test-neural-network-models-in-one-function-call-eb6f1fa5019d). The innovation and major benefit is that you no longer
need to write additional testing code for your model training. Just add a few
lines of code specifying the checks before training, torcheck will then take over and
perform the checks simultaneouly while the training happens.

Another benefit is that torcheck allows you to check your model on different levels.
Instead of checking the whole model, you can specify checks for a submodule, a linear
layer, or even the weight tensor! This enables more customization around the sanity
checks.

## Installation
```
pip install torcheck
```

## Torcheck in 5 minutes
OK, suppose you have coded up a standard PyTorch training routine like this:
```
model = Model()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

# torcheck code goes here

for epoch in range(num_epochs):
    for x, y in dataloader:
        # calculate loss and backward propagation
```

By simply adding a few lines of code right before the for loop, you can be more confident
about whether your model is training as expected!

### Step 1: Registering your optimizer(s)
First, register the optimizer(s) with torcheck:
```
torcheck.register(optimizer)
```

### Step 2: Adding sanity checks
Torcheck enables you to perform a wide range of checks, on both module level and tensor
level.

A rule of thumb is that use APIs with `add_module` prefix when checking something that
subclasses from `nn.Module`, use APIs with `add_tensor` prefix when checking tensors.

#### Parameters change/not change
You can check whether model parameters actually get updated during the training.
Or you can check whether they remain constant if you want them to be frozen.

For our example, some of the possible checks are:

```
# check all the model parameters will change
# module_name is optional, but it makes error messages more informative when checks fail
torcheck.add_module_changing_check(model, module_name="my_model")
```

```
# check the linear layer's parameters won't change
torcheck.add_module_unchanging_check(model.linear_0, module_name="linear_layer_0")
```

```
# check the linear layer's weight parameters will change
torcheck.add_tensor_changing_check(
    model.linear_0.weight, tensor_name="linear_0.weight", module_name="my_model"
)
```

```
# check the linear layer's bias parameters won't change
torcheck.add_tensor_unchanging_check(
    model.linear_0.bias, tensor_name="linear_0.bias", module_name="my_model"
)
```

#### Output range check
The basic use case is that you can check whether model outputs are all within a range,
say (-1, 1).

You can also check that model outputs are not all within a range. This is useful when
you want softmax to behave correctly. It enables you to check model ouputs are not all
within (0, 1).

You can check the final model output or intermediate output of a submodule.
```
# check model outputs are within (-1, 1)
torcheck.add_module_output_range_check(
    model, output_range=(-1, 1), module_name="my_model"
)
```

```
# check outputs from the linear layer are within (-5, 5)
torcheck.add_module_output_range_check(
    model.linear_0, output_range=(-5, 5), module_name="linear_layer_0"
)

```

```
# check model outputs are not all within (0, 1)
# aka softmax hasn't been applied before loss calculation
torcheck.add_module_output_range_check(
    model,
    output_range=(0, 1),
    negate_range=True,
    module_name="my_model",
)
```

#### NaN check
Check whether parameters become NaN during training, or model outputs contain NaN.

```
# check whether model parameters become NaN or outputs contain NaN
torcheck.add_module_nan_check(model, module_name="my_model")
```

```
# check whether linear layer's weight parameters become NaN
torcheck.add_tensor_nan_check(
    model.linear_0.weight, tensor_name="linear_0.weight", module_name="my_model"
)
```

#### Inf check
Check whether parameters become infinite (positive or negative infinity) during training,
or model outputs contain infinite value.

```
# check whether model parameters become infinite or outputs contain infinite value
torcheck.add_module_inf_check(model, module_name="my_model")
```

```
# check whether linear layer's weight parameters become infinite
torcheck.add_tensor_inf_check(
    model.linear_0.weight, tensor_name="linear_0.weight", module_name="my_model"
)
```

#### Adding multiple checks in one call
You can add all checks for a module/tensor in one call:
```
# add all checks for model together
torcheck.add_module(
    model,
    module_name="my_model",
    changing=True,
    output_range=(-1, 1),
    check_nan=True,
    check_inf=True,
)
```

```
# add all checks for linear layer's weight together
torcheck.add_tensor(
    model.linear_0.weight,
    tensor_name="linear_0.weight",
    module_name="my_model",
    changing=True,
    check_nan=True,
    check_inf=True,
)
```

### Step 3: Training and fixing
After adding all the checks, run the training as usual and fix errors if any.

By default torcheck's error messages don't include tensor value information. If you
think it would be helpful, you can add the following line inside your torcheck code:
```
torcheck.verbose_on()
```

You can turn it off again by calling
```
torcheck.verbose_off()
```

### (Optional) Step 4: Turning off checks
When your model has passed all the checks, you can easily turn them off by calling
```
torcheck.disable()
```
This is useful when you want to run your model on a validation set, or you just want to
remove the checking overhead from training.

If you want to turn on the checks again, just call
```
torcheck.enable()
```
