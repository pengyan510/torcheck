from dataclasses import dataclass
from typing import Union
import warnings

import torch

from .utils import message_utils


@dataclass
class OutputSpec:
    module_name: str = None
    range: Union[list, tuple] = None
    negate: bool = False
    check_nan: bool = False
    check_inf: bool = False

    @property
    def name(self):
        if self.module_name is None:
            return "Module's output"
        else:
            return f"Module {self.module_name}'s output"

    @property
    def condition(self):
        low, high = self.range
        if low is None:
            return f"< {high}"
        elif high is None:
            return f"> {low}"
        else:
            return f"> {low} and < {high}"

    def update(
        self,
        module_name=None,
        range=None,
        negate=False,
        check_nan=False,
        check_inf=False,
    ):
        if module_name is not None and module_name != self.module_name:
            old_name = self.name
            self.module_name = module_name
            warnings.warn(f"{old_name} is renamed as {self.name}.")
        if range is not None:
            self.range = range
            self.negate = negate
        if check_nan:
            self.check_nan = True
        if check_inf:
            self.check_inf = True

    def validate(self, output):
        error_items = []
        if self.range is not None:
            error_items.append(self.validate_range(output))
        if self.check_nan:
            error_items.append(self.validate_nan(output))
        if self.check_inf:
            error_items.append(self.validate_inf(output))

        error_items = [_ for _ in error_items if _ is not None]
        if len(error_items):
            raise RuntimeError(message_utils.make_message(error_items, output))

    def validate_range(self, output):
        low, high = self.range
        status = torch.ones_like(output, dtype=torch.bool)
        if low is not None:
            status = output >= low
        if high is not None:
            status = status & (output <= high)

        if not self.negate:
            if not torch.all(status).item():
                return (
                    f"{self.name} should all {self.condition}. " "Some are out of range"
                )
        else:
            if torch.all(status).item():
                return f"{self.name} shouldn't all {self.condition}"

    def validate_nan(self, output):
        if torch.any(torch.isnan(output)).item():
            return f"{self.name} contains NaN."

    def validate_inf(self, output):
        if torch.any(torch.isinf(output)).item():
            return f"{self.name} contains inf."
