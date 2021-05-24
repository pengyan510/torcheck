from dataclasses import dataclass, field
from collections import defaultdict

import torch.nn as nn

from .spec import SpecList


@dataclass
class Registry:
    optimizer_to_spec: dict = field(default_factory=lambda: defaultdict(SpecList), init=False)
    tensor_to_optimizer: dict = field(default_factory=dict, init=False)
    active_optimizers: set = field(default_factory=set, init=False)

    def run_tests(self, optimizer):
        
        def decorator(func):
            
            def inner(*args, **kwargs):
                func(*args, **kwargs)
                if optimizer in self.active_optimizers:
                    self.optimizer_to_spec[optimizer].validate()

            return inner

        return decorator

    def register(self, optimizer):
        optimizer.step = self.run_tests(optimizer)(optimizer.step)
        for param_group in optimizer.param_groups:
            for param in param_group["params"]:
                self.tensor_to_optimizer[param] = optimizer
        self.active_optimizers.add(optimizer)
    
    def add_tensor(
        self,
        tensor,
        tensor_name,
        module_name=None,
        changing=None,
        max=None,
        min=None
    ):
        optimizer = self.tensor_to_optimizer.get(tensor, None)
        if optimizer is None:
            raise RuntimeError(
                "The tensor doesn't belong to any optimizer. "
                "Please register the optimizer first."
            )
        self.optimizer_to_spec[optimizer].add(
            tensor=tensor,
            tensor_name=tensor_name,
            module_name=module_name,
            changing=changing,
            max=max,
            min=min
        )

    def add_module(
        self,
        module,
        module_name=None,
        changing=None,
        max=None,
        min=None
    ):
        if not isinstance(module, nn.Module):
            raise RuntimeError(
                f"Module should be nn.Module type, but is {type(module)}."
            )
        for name, param in module.named_parameters():
            self.add_tensor(
                tensor=param,
                tensor_name=name,
                module_name=module_name,
                changing=changing,
                max=max,
                min=min
            ) 

    def add_changing_module(self, module, module_name=None):
        self.add_module(
            module,
            module_name=module_name,
            changing=True
        )

    def disable(self, optimizers=None):
        if optimizers is None:
            optimizers = self.optimizer_to_spec.keys() 
        for optimizer in optimizers:
            self.active_optimizers.remove(optimizer)

    def enable(self, optimizers=None):
        if optimizers is None:
            optimizers = self.optimizer_to_spec.keys()
        for optimizer in optimizers:
            self.active_optimizers.add(optimizer)

