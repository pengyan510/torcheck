from dataclasses import dataclass, field
import warnings

import torch

from .utils import message_utils


@dataclass
class SpecItem:
    tensor: torch.Tensor
    tensor_name: str
    module_name: str = None
    changing: bool = None
    check_nan: bool = False
    check_inf: bool = False
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

    def update(
        self,
        tensor_name,
        module_name=None,
        changing=None,
        check_nan=False,
        check_inf=False,
    ):
        if (tensor_name != self.tensor_name) or (
            module_name is not None and module_name != self.module_name
        ):
            old_name = self.name
            self.tensor_name = tensor_name
            if module_name is not None:
                self.module_name = module_name
            warnings.warn(f"{old_name} is renamed as {self.name}")
        if changing is not None:
            self.changing = changing
        if check_nan:
            self.check_nan = True
        if check_inf:
            self.check_inf = True

    def validate(self):
        error_items = []
        if self.changing is not None:
            error_items.append(self.validate_changing())
        if self.check_nan:
            error_items.append(self.validate_nan())
        if self.check_inf:
            error_items.append(self.validate_inf())

        error_items = [_ for _ in error_items if _ is not None]
        return message_utils.make_message(error_items, self.tensor)

    def validate_changing(self):
        if self.changing:
            if torch.equal(self.tensor, self._old_copy):
                return f"{self.name} should change."
        else:
            if not torch.equal(self.tensor, self._old_copy):
                return f"{self.name} should not change."

        self._old_copy = self.tensor.detach().clone()

    def validate_nan(self):
        if torch.any(torch.isnan(self.tensor)).item():
            return f"{self.name} contains NaN."

    def validate_inf(self):
        if torch.any(torch.isinf(self.tensor)).item():
            return f"{self.name} contains inf."


@dataclass
class ParamSpec:
    specs: dict = field(default_factory=dict)

    def add(
        self,
        tensor,
        tensor_name,
        module_name=None,
        changing=None,
        check_nan=False,
        check_inf=False,
    ):
        if tensor in self.specs:
            self.specs[tensor].update(
                tensor_name=tensor_name,
                module_name=module_name,
                changing=changing,
                check_nan=check_nan,
                check_inf=check_inf,
            )
        else:
            self.specs[tensor] = SpecItem(
                tensor=tensor,
                tensor_name=tensor_name,
                module_name=module_name,
                changing=changing,
                check_nan=check_nan,
                check_inf=check_inf,
            )

    def validate(self):
        error_strings = []
        for spec in self.specs.values():
            error_string = spec.validate()
            if len(error_string) > 0:
                error_strings.append(error_string)
        if len(error_strings) > 0:
            error_msg = "\n".join(error_strings)
            raise RuntimeError(
                f"The following errors are detected while training:\n{error_msg}"
            )
