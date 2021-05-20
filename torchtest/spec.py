from dataclasses import dataclass, field

import torch


@dataclass
class SpecItem:
    tensor: torch.Tensor
    tensor_name: str
    module_name: str = None
    changing: bool = None
    max: float = None
    min: float = None
    _old_copy: torch.Tensor = field(init=False, default=None)

    def __post_init__(self):
        if self.changing is not None:
            self._old_copy = self.tensor.detach().clone()

    @property
    def name(self):
        if self.module_name is None:
            return self.tensor_name
        else:
            return f"Module {self.module_name}'s {self.tensor_name}"

    def validate(self):
        if self.changing is not None:
            self.validate_changing()
        if self.max is not None:
            self.validate_max()
        if self.min is not None:
            self.validate_min()

    def validate_changing(self):
        if self.changing:
            if torch.equal(self.tensor, self._old_copy):
                raise RuntimeError(f"{self.name} should change.")
        else:
            if not torch.equal(self.tensor, self._old_copy):
                raise RuntimeError(f"{self.name} should not change.")

        self._old_copy = self.tensor.detach().clone()

    def validate_max(self):
        pass

    def validate_min(self):
        pass


@dataclass
class SpecList:
    specs: list = field(default_factory=list)

    def add(
        self,
        tensor,
        tensor_name,
        module_name=None,
        changing=None,
        max=None,
        min=None
    ):
        self.specs.append(
            SpecItem(
                tensor=tensor,
                tensor_name=tensor_name,
                module_name=module_name,
                changing=changing,
                max=max,
                min=min
            )
        )

    def validate(self):
        for spec in self.specs:
            spec.validate()

