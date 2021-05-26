from dataclasses import dataclass, field

import torch


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

    def validate(self):
        error_items = []
        if self.changing is not None:
            error_items.append(self.validate_changing())
        if self.check_nan:
            error_items.append(self.validate_nan())
        if self.check_inf:
            error_items.append(self.validate_inf())

        return message_utils.make_message(error_items, self.tensor)

    def validate_changing(self):
        if self.changing:
            if torch.equal(self.tensor, self._old_copy):
                return f"{self.name} should change."
        else:
            if not torch.equal(self.tensor, self._old_copy):
                return f"{self.name} should not change."

        self._old_copy = self.tensor.detach().clone()

    def validate_nan():
        if torch.any(torch.isnan(self.tensor)).item():
            return f"{self.name} contains NaN." 

    def validate_inf():
        if torch.any(torch.isinf(self.tensor)).item():
            return f"{self.name} contains inf." 


@dataclass
class ParamSpec:
    specs: list = field(default_factory=list)

    def add(
        self,
        tensor,
        tensor_name,
        module_name=None,
        changing=None,
        check_nan=False,
        check_inf=False
    ):
        self.specs.append(
            SpecItem(
                tensor=tensor,
                tensor_name=tensor_name,
                module_name=module_name,
                changing=changing,
                check_nan=check_nan,
                check_inf=check_inf
            )
        )

    def validate(self):
        error_strings = []
        for spec in self.specs:
            error_string = spec.validate()
            if len(error_string) > 0:
                error_strings.append(error_string)
        if len(error_strings) > 0:
            error_msg = "\n".join(error_strings)
            raise RuntimeError(
                f"The following errors are detected while training:\n{error_msg}"
            )

