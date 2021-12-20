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
            'max': lambda x: torch.max(x)/10,
            'min': lambda x: torch.min(x)*10,
            'mean': torch.mean
        }
        self.downsample = downsample

    def _do(self, x, fn):
        f = fn(x, dim=2)
        f = f if fn == torch.mean else f[0]
        return x/x * f.unsqueeze(2)

    def _noise(self, x):
        return x/x * torch.randn_like(x)/10

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
        return x/x

    def _downsample(self, x):
        h, w = x.shape[-2], x.shape[-1]
        return x[:, :, :h//2, :w//2] # TODO: Replace with proper calculations of output size from stride and padding

    def forward(self, x, edge):

        if self.type in self.fns.keys():
            x = self._do(x, self.fns[self.type])
        elif self.type == 'zero':
            x = self._zero(x)
        elif self.type == 'ones':
            x = self._ones(x)
        else:
            x = self._noise(x)

        x = self._expand(x)

        if self.downsample:
            x = self._downsample(x)

        return x

    def get_embedded_ops(self):
        return None
