"""Theoretical privacy accounting for DP-FedAvg with Poisson client sampling."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

RDP_ORDERS = [1 + x / 10 for x in range(1, 100)] + list(range(12, 64))


def rdp_epsilon(z: float, q: float, rounds: int, delta: float) -> float:
    """(eps, delta) from the Renyi DP accountant (Opacus). This is what the notebook reports."""
    if z <= 0:
        return float("inf")
    from opacus.accountants.analysis import rdp as rdp_an

    r = rdp_an.compute_rdp(q=q, noise_multiplier=z, steps=rounds, orders=RDP_ORDERS)
    eps, _ = rdp_an.get_privacy_spent(orders=RDP_ORDERS, rdp=r, delta=delta)
    return float(eps)


def gdp_delta(eps: float, mu: float) -> float:
    """delta(eps) for a mu-Gaussian-DP mechanism (Dong, Roth and Su 2019)."""
    return norm.cdf(-eps / mu + mu / 2) - math.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def gdp_epsilon(mu: float, delta: float) -> float:
    """Smallest eps such that a mu-GDP mechanism is (eps, delta)-DP."""
    if not np.isfinite(mu):
        return float("inf")
    if mu <= 0 or gdp_delta(0.0, mu) <= delta:
        return 0.0
    hi = 1.0
    while gdp_delta(hi, mu) > delta:
        hi *= 2
    return float(brentq(lambda e: gdp_delta(e, mu) - delta, 0.0, hi, xtol=1e-6))


def clt_gdp_mu(z: float, q: float, rounds: int) -> float:
    """Central-limit GDP approximation for subsampled Gaussian composition (Bu et al. 2020).

    Approximate, not a rigorous bound, but usually much tighter than RDP.
    """
    if z <= 0:
        return float("inf")
    return q * math.sqrt(rounds * (math.exp(1 / z**2) - 1))


def clt_epsilon(z: float, q: float, rounds: int, delta: float) -> float:
    return gdp_epsilon(clt_gdp_mu(z, q, rounds), delta)
