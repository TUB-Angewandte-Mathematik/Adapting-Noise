import math
from typing import Optional

import torch
import torch.nn as nn


class TorchWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self._labels: Optional[torch.Tensor] = None

    def forward(self,x,t):
       
        time = t.repeat(x.shape[0])[:, None].to(device=x.device, dtype=x.dtype)
        labels = self._labels
        if labels is not None:
            if labels.shape[0] != x.shape[0]:
                raise ValueError("Mismatch between stored labels and batch size in TorchWrapper")
            out = self.model(time, x, labels)
        else:
            out = self.model(time, x)
            
        return out

    def set_labels(self, labels: Optional[torch.Tensor]) -> None:
        self._labels = labels


class VelocityFieldAdapter(nn.Module):
    """Adapts image-space models (e.g. UNets) to the (t, x[, labels]) API."""

    def __init__(self, model: nn.Module, image_shape):
        super().__init__()
        self.model = model
        self.image_shape = tuple(image_shape)
        self._flat_dim = math.prod(self.image_shape)

    def forward(self, t: torch.Tensor, x: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch = x.shape[0]
        x_reshaped = x.reshape(batch, *self.image_shape)
        timesteps = t.view(batch).to(x_reshaped.device)
        if labels is not None:
            labels = labels.to(device=x_reshaped.device, dtype=torch.long)
            out = self.model(x_reshaped, timesteps, y=labels)
        else:
            out = self.model(x_reshaped, timesteps)
        return out.reshape(batch, self._flat_dim)
    
class ODEWrapper(torch.nn.Module):
    def __init__(self, fmap):
        super().__init__()
        self.fmap = fmap
        self.nfe = 0

    def reset_nfe(self):
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        return self.fmap(x, t)


class PotentialGradWrapper(nn.Module):
    """
    Wraps a scalar potential s(t, x) and exposes its spatial gradient ∇_x s(t, x)
    so it can be used as an ODE drift field.
    Expects `potential` to implement forward(time, x) -> (B, 1).
    """
    def __init__(self, potential: nn.Module):
        super().__init__()
        self.potential = potential

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Ensure proper shapes
        if t.ndim == 0:
            t_in = t.repeat(x.shape[0])[:, None].float()
        else:
            t_in = t

        # Compute gradient w.r.t. x only; do not build large graphs
        with torch.enable_grad():
            x_req = x.detach().requires_grad_(True)
            s = self.potential(t_in, x_req)  # (B, 1)
            grad_x = torch.autograd.grad(s.sum(), x_req, create_graph=False, retain_graph=False)[0]
        return grad_x
