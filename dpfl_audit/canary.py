"""Client-level auditing with random-direction canary clients.

Follows the idea of Andrew et al. 2024, "One-shot Empirical Privacy Estimation for
Federated Learning". A canary client ignores the data and always submits the same
update: a fixed random unit direction scaled to the clipping norm. That is the
strongest update a client is allowed to send. The auditor then checks how strongly
each canary direction shows up in what the server released, and compares inserted
canaries against directions that were never inserted.

In ~67k dimensions random directions are almost orthogonal, so the cosine of an
unrelated direction with the model delta is close to N(0, 1/d). Inserted canaries
shift that distribution. If the shift is mu standard deviations, the mechanism looks
like mu-Gaussian-DP to this auditor, which converts to an (eps, delta) estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .accounting import gdp_epsilon
from .attacks import eps_lower_bound


@dataclass
class CanarySet:
    inserted: torch.Tensor  # [m, d] unit vectors that take part in training
    held_out: torch.Tensor  # [m0, d] unit vectors that never do

    @classmethod
    def make(cls, d: int, n_inserted: int, n_held_out: int, seed: int, device) -> "CanarySet":
        g = torch.Generator(device="cpu").manual_seed(10_000 + seed)
        v = torch.randn(n_inserted + n_held_out, d, generator=g)
        v = v / v.norm(dim=1, keepdim=True)
        v = v.to(device)
        return cls(inserted=v[:n_inserted], held_out=v[n_inserted:])

    @property
    def all(self) -> torch.Tensor:
        return torch.cat([self.inserted, self.held_out])


@dataclass
class CanaryResult:
    final_obs: np.ndarray  # cosine(final model delta, canary) for inserted canaries
    final_null: np.ndarray
    iter_obs: np.ndarray  # max over rounds of cosine(round update, canary)
    iter_null: np.ndarray

    def estimate(self, delta: float) -> dict:
        out = {}
        for tag, obs, null in [
            ("final", self.final_obs, self.final_null),
            ("iterates", self.iter_obs, self.iter_null),
        ]:
            mu = float((obs.mean() - null.mean()) / null.std(ddof=1))
            out[f"{tag}_mu"] = mu
            out[f"{tag}_eps_est"] = gdp_epsilon(mu, delta)
            out[f"{tag}_eps_lb"] = eps_lower_bound(obs, null, delta=delta, fp_grid=(1, 2, 5, 10, 20, 50))
        return out
