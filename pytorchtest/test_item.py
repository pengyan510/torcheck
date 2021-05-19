from dataclasses import dataclass, field


@dataclass
class TestItem:
    layer: nn.Module # not sure
    changing: bool = None
    max: float = None
    min: float = None
    _layer_copy: nn.Module = field(init=False, default=None)

    def validate(self):

    def validate_changing(self):

    def validate_max(self):

    def validate_min(self):
