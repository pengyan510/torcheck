from dataclasses import dataclass
from typing import Union

import torch


@dataclass
class OutputSpec:
    range: Union[list, tuple]
    negate: bool = False
    module_name: str = None

    @property
    def name(self):
        if module_name is None:
            return "Module's output"
        else:
            return f"{self.module_name}'s output"

    @property
    def condition(self):
        low, high = self.range
        if low is None:
            return f"< {high}"
        elif high is None:
            return f"> {low}"
        else:
            return f"> {low} and < {high}"

    def validate(self, output):
        low, high = self.range
        status = torch.ones_like(output, dtype=torch.bool)
        if low is not None:
            status = output >= low
        if high is not None:
            status = status & (output <= high)
        
        if not negate:
            if not torch.all(status).item():
                raise RuntimeError(
                    f"{self.name} should all {self.condition}. Some are out of range"
                )
        else:
            if torch.all(status).item():
                raise RuntimeError(
                    f"{self.name} shouldn't all {self.condition}"
                )

