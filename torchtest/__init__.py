from .registry import Registry


registry = Registry()
register = registry.register
add_tensor = registry.add_tensor
add_module = registry.add_module
add_changing_module = registry.add_changing_module
disable = registry.disable
enable = registry.enable

