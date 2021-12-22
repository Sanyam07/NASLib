import torch
from torch.nn.modules import module

from ..core.primitives import AbstractPrimitive


class DestroySignal(AbstractPrimitive):
    """
    Module which destroys the incoming signal in one of six ways (replace with channels
    with noise, zeros, ones, min, max and mean).
    """

    def __init__(self, C_out, module_type='max', downsample=False):
        super().__init__(locals())
        self.C_out = C_out
        self.module_type = module_type
        self.fns = {
            'min': lambda x: torch.min(x, dim=2)[0]*10,
            'max': lambda x: torch.max(x, dim=2)[0]/10,
            'mean': lambda x: torch.mean(x, dim=2),
            'var': lambda x: torch.var(x, dim=2)
        }
        self.downsample = downsample
        self.epsilon = 10e-5

    def _do(self, x, fn):
        f = fn(x)
        return x/(x + self.epsilon) * f.unsqueeze(2)

    def _noise(self, x):
        return x/(x + self.epsilon) * torch.randn_like(x)/10

    def _expand(self, x):
        n_channels = x.shape[-3]
        channels = []

        for i in range(self.C_out//n_channels):
            channels.append(x)

        channels.append(x[:, :self.C_out%n_channels, :, :])

        return torch.cat(channels, dim=1)

    def _zero(self, x):
        return x * torch.zeros_like(x)

    def _ones(self, x):
        return x/(x + self.epsilon)

    def _downsample(self, x):
        h, w = x.shape[-2], x.shape[-1]
        return x[:, :, :h//2, :w//2] # TODO: Replace with proper calculations of output size from stride and padding

    def forward(self, x, edge_data):

        if self.module_type in self.fns.keys():
            x = self._do(x, self.fns[self.module_type])
        elif self.module_type == 'zero':
            x = self._zero(x)
        elif self.module_type == 'ones':
            x = self._ones(x)
        else:
            raise NotImplementedError(f'Operation type "{self.module_type}" not supported')

        x = self._expand(x)

        if self.downsample:
            x = self._downsample(x)

        return x

    def get_embedded_ops(self):
        return None
