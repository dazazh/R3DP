import torch.nn as nn
import torch

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        # self._dummy_variable = nn.Parameter()
        self._dummy_variable = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
