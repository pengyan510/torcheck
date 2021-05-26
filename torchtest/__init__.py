from .registry import Registry
from .message_utils import (
    verbose_on,
    verbose_off,
    is_verbose
)

registry = Registry()

register = registry.register
add_tensor = registry.add_tensor
add_module = registry.add_module
add_changing_check = registry.add_changing_check
add_not_changing_check = registry.add_not_changing_check
add_output_range_check = registry.add_output_range_check

disable_optimizers = registry.disable_optimizers
disable_modules = registry.disable_modules
disable = registry.disable
enable_optimizers = registry.enable_optimizers
enable_modules = registry.enable_modules
enable = registry.enable

