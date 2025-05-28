
import numpy as np
from __main__ import AutoRegisteredScoringModuleBase # Assuming base class is accessible

class MyFeatureInPackage(AutoRegisteredScoringModuleBase):
    def get_output(self) -> np.ndarray:
        return self.grid * 100 + 42

# A class that returns a list, to test auto-correction (type coercion)
class ListReturnerModule:
    _is_autoregister_module_via_decorator = True # Example marker
    def __init__(self, grid): self.grid = grid
    def get_output(self): # Returns list
        return (self.grid.flatten() / 2).tolist() 
