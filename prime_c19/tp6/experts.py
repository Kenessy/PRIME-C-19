"""Expert routing utilities for TP6.

The original kernel uses pointer bin addresses to choose between multiple output
heads (experts). This module contains the extracted router.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LocationExpertRouter(nn.Module):
    """Route each batch element to one of N output heads.

    Routing rule (behavior-preserving):
      - If `pointer_addresses` is None, all samples route to expert 0.
      - Else `expert_index = pointer_addresses % num_experts`.

    This matches `tournament_phase6.LocationExpertRouter`.
    """

    def __init__(self, d_model: int, vocab_size: int, num_experts: int = 1):
        super().__init__()
        self.num_experts = max(1, int(num_experts))
        self.in_features = int(d_model)
        self.out_features = int(vocab_size)
        if self.num_experts == 1:
            self.single = nn.Linear(d_model, vocab_size)
            self.experts = None
        else:
            self.single = None
            self.experts = nn.ModuleList([nn.Linear(d_model, vocab_size) for _ in range(self.num_experts)])

    def reset_parameters(self):
        def init_layer(layer):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        if self.single is not None:
            init_layer(self.single)
        else:
            for expert in self.experts:
                init_layer(expert)

    def forward(self, x: torch.Tensor, pointer_addresses: torch.Tensor | None = None) -> torch.Tensor:
        if self.single is not None:
            return self.single(x)

        if pointer_addresses is None:
            expert_indices = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        else:
            expert_indices = pointer_addresses.to(torch.long, non_blocking=True) % self.num_experts

        out_dtype = self.experts[0].weight.dtype
        out = torch.zeros(x.shape[0], self.experts[0].out_features, device=x.device, dtype=out_dtype)
        for i, expert in enumerate(self.experts):
            mask = expert_indices == i
            if mask.any():
                out[mask] = expert(x[mask]).to(out_dtype)
        return out
