"""Membership inference attacks and statistically valid empirical-epsilon lower bounds.

Convention everywhere: a higher score means "more likely a member".
"""

from __future__ import annotations

import numpy as np
from scipy.stats import beta as beta_dist
from sklearn.metrics import roc_auc_score, roc_curve


# --------------------------------------------------------------------------- scores
def loss_score(loss: np.ndarray) -> np.ndarray:
    """Yeom et al. 2018: low loss means member."""
    return -loss


def calibrated_loss_score(target_loss: np.ndarray, ref_losses: np.ndarray) -> np.ndarray:
    """Watson et al. 2022 difficulty calibration: how much lower is the loss than on models
    that never saw this example? ref_losses has shape [R, n]."""
    return ref_losses.mean(0) - target_loss


def lira_offline_score(target_phi: np.ndarray, ref_phis: np.ndarray) -> np.ndarray:
    """Offline LiRA (Carlini et al. 2022) with a global variance.

    phi is the logit-scaled confidence on the true class. Reference models are all OUT
    models, so the score is how many standard deviations the target's confidence sits
    above the per-example OUT distribution.
    """
    mu_out = ref_phis.mean(0)
    if ref_phis.shape[0] > 1:
        sigma = float(np.std(ref_phis - mu_out, ddof=1))
    else:
        sigma = float(np.std(ref_phis))
    return (target_phi - mu_out) / max(sigma, 1e-8)


# --------------------------------------------------------------------------- metrics
def _tpr_at(fpr: np.ndarray, tpr: np.ndarray, a: float) -> float:
    return float(tpr[np.searchsorted(fpr, a, side="right") - 1])


def mia_metrics(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict:
    s = np.concatenate([member_scores, nonmember_scores])
    y = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    fpr, tpr, _ = roc_curve(y, s)
    return dict(
        auc=float(roc_auc_score(y, s)),
        adv=float((tpr - fpr).max()),
        tpr_01=_tpr_at(fpr, tpr, 0.001),
        tpr_1=_tpr_at(fpr, tpr, 0.01),
        tpr_10=_tpr_at(fpr, tpr, 0.10),
    )


def roc(member_scores, nonmember_scores):
    s = np.concatenate([member_scores, nonmember_scores])
    y = np.r_[np.ones(len(member_scores)), np.zeros(len(nonmember_scores))]
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr


# --------------------------------------------------------------------------- auditing
def cp_upper(k: int, n: int, a: float) -> float:
    return 1.0 if k >= n else float(beta_dist.ppf(1 - a, k + 1, n - k))


def cp_lower(k: int, n: int, a: float) -> float:
    return 0.0 if k <= 0 else float(beta_dist.ppf(a, k, n - k + 1))


def eps_lower_bound(
    member_scores: np.ndarray,
    nonmember_scores: np.ndarray,
    delta: float = 1e-5,
    conf: float = 0.05,
    fp_grid=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000),
) -> float:
    """Empirical epsilon lower bound that holds with probability >= 1 - conf.

    (eps, delta)-DP implies TPR <= e^eps * FPR + delta for every test, so
    eps >= ln((TPR - delta) / FPR). We use a one-sided Clopper-Pearson lower bound on
    TPR, an upper bound on FPR, and a Bonferroni correction over the threshold grid.
    """
    sp, sn = np.asarray(member_scores), np.asarray(nonmember_scores)
    grid = [g for g in fp_grid if g < len(sn)]
    a = conf / max(len(grid), 1)
    sn_sorted = np.sort(sn)[::-1]
    best = 0.0
    for fp in grid:
        thr = sn_sorted[fp]
        fpc, tpc = int((sn > thr).sum()), int((sp > thr).sum())
        tpr_lo, fpr_up = cp_lower(tpc, len(sp), a), cp_upper(fpc, len(sn), a)
        if tpr_lo > delta:
            best = max(best, float(np.log((tpr_lo - delta) / fpr_up)))
    return best
