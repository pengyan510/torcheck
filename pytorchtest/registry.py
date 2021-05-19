from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Registry:
    optimizer_tests: dict = field(default_factory=defaultdict(list), init=False)

    def run_tests(self, optimizer):
        
        def decorator(func):
            
            def inner():
                func()
                self.optimizer_tests[optimizer].validate()

            return inner

        return decorator

    def register(self, optimizer):
        optimizer.step = self.run_tests(optimizer)(optimizer.step)
    
    def add_test(self, layer, changing=None, max=None, min=None):

    def add_changing_test(self, layer):
        self.add_test(layer, changing=True) 
