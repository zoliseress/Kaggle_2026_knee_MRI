"""Exponential moving average of the model weights.

One run's weights wander between optimizer steps; averaging them over the last few
epochs removes part of that noise - the same variance a seed ensemble removes, inside a
single model. The averaged copy has the model's own state_dict keys, so a checkpoint
that stores it under "model" loads everywhere a raw model would.

    ema_t = d_t * ema_{t-1} + (1 - d_t) * param_t,   d_t = min(decay, (1 + t) / (10 + t))

The warmup in d_t keeps the first, nearly pretrained weights from lingering: after the
first update d_1 = 2/11, so the average follows the model closely until it has seen
enough steps for the nominal decay to apply.
"""

from __future__ import annotations

import copy

import torch


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        if not 0.0 < float(decay) < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        self.decay = float(decay)
        self.updates = 0
        self.module = copy.deepcopy(model).eval()
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    def current_decay(self) -> float:
        """The decay the next update will use."""
        t = self.updates + 1
        return min(self.decay, (1.0 + t) / (10.0 + t))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Average the parameters; copy the buffers (BatchNorm running stats are frozen here)."""
        d = self.current_decay()
        ema_params = dict(self.module.named_parameters())
        for name, parameter in model.named_parameters():
            ema_params[name].mul_(d).add_(parameter.detach().to(ema_params[name].dtype), alpha=1.0 - d)
        ema_buffers = dict(self.module.named_buffers())
        for name, buffer in model.named_buffers():
            ema_buffers[name].copy_(buffer)
        self.updates += 1

    @torch.no_grad()
    def reset(self, model: torch.nn.Module) -> None:
        """Restart the average from `model` (e.g. after loading initial weights)."""
        self.module.load_state_dict(model.state_dict())
        self.updates = 0

    def state_dict(self) -> dict:
        return self.module.state_dict()

    def load_state_dict(self, state: dict, updates: int) -> None:
        self.module.load_state_dict(state)
        self.updates = int(updates)
