import math

import numpy as np
import pytest
import torch

from dpfl_audit.accounting import clt_epsilon, gdp_delta, gdp_epsilon, rdp_epsilon
from dpfl_audit.attacks import eps_lower_bound, lira_offline_score, mia_metrics
from dpfl_audit.fl import ParamLayout, VmapTrainer, local_update_sequential
from dpfl_audit.model import CNN

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_param_count():
    assert ParamLayout(CNN()).d == 66_890


def test_vmap_matches_sequential_loop():
    """Vectorised training must produce the same client updates as the notebook's loop.

    TF32 is switched off: cuDNN picks different conv kernels for the batched path and
    TF32 rounding (~1e-3 per op) would otherwise compound over the SGD steps."""
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    model = CNN().to(DEV)
    tr = VmapTrainer(model)
    g = tr.layout.flatten(model).clone()
    K, n, epochs, batch, lr = 6, 20, 3, 10, 0.05
    X = torch.randn(K, n, 3, 32, 32, device=DEV)
    Y = torch.randint(0, 10, (K, n), device=DEV)
    perms = torch.stack([torch.stack([torch.randperm(n, device=DEV) for _ in range(K)]) for _ in range(epochs)])

    fast = tr.local_updates(g, X, Y, epochs, batch, lr, perms=perms)
    slow = torch.stack(
        [local_update_sequential(CNN().to(DEV), tr.layout, g, X[k], Y[k], epochs, batch, lr, perms[:, k]) for k in range(K)]
    )
    rel = (fast - slow).norm() / slow.norm()
    assert rel < 1e-3, rel


@pytest.mark.parametrize("z,expected", [(0.5, 35.23), (1.0, 7.90), (2.0, 2.58), (4.0, 1.08)])
def test_rdp_matches_notebook(z, expected):
    assert rdp_epsilon(z, q=0.1, rounds=100, delta=1e-5) == pytest.approx(expected, abs=0.01)


def test_gdp_epsilon_inverts_delta():
    for mu in [0.3, 1.0, 2.5]:
        eps = gdp_epsilon(mu, 1e-5)
        assert gdp_delta(eps, mu) == pytest.approx(1e-5, rel=1e-3)


def test_clt_is_tighter_than_rdp():
    assert clt_epsilon(1.0, 0.1, 100, 1e-5) < rdp_epsilon(1.0, 0.1, 100, 1e-5)


def test_no_leak_gives_no_bound():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=10_000), rng.normal(size=10_000)
    assert eps_lower_bound(a, b) == 0.0
    assert mia_metrics(a, b)["auc"] == pytest.approx(0.5, abs=0.02)


def test_bound_never_exceeds_true_epsilon():
    """Scores from a Gaussian mechanism with known mu: the audit must stay below truth."""
    rng = np.random.default_rng(1)
    mu = 3.0
    lb = eps_lower_bound(rng.normal(mu, 1, 10_000), rng.normal(0, 1, 10_000))
    assert 0 < lb < gdp_epsilon(mu, 1e-5)


def test_lira_calibration_removes_per_example_difficulty():
    rng = np.random.default_rng(2)
    difficulty = rng.normal(0, 3, 5000)
    refs = difficulty + rng.normal(0, 0.5, (4, 5000))
    members = difficulty[:2500] + 0.5 + rng.normal(0, 0.5, 2500)
    nonmembers = difficulty[2500:] + rng.normal(0, 0.5, 2500)
    raw = mia_metrics(members, nonmembers)["auc"]
    cal = mia_metrics(lira_offline_score(members, refs[:, :2500]), lira_offline_score(nonmembers, refs[:, 2500:]))["auc"]
    assert cal > raw + 0.1
