"""DP-FedAvg with vectorised client training.

The notebook trains each sampled client one after another with an nn.Module and an
optimizer. Here every sampled client in a round is trained at once: the model is
called functionally with torch.func, per-client gradients come from vmap(grad(...)),
and plain SGD is applied to a stacked [K, ...] copy of the parameters. Same maths,
one batched kernel launch per layer instead of K.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, vmap
from sklearn.metrics import roc_auc_score

from .canary import CanaryResult, CanarySet
from .config import FLConfig
from .model import CNN


# --------------------------------------------------------------------------- params
class ParamLayout:
    """Converts between a flat parameter vector and a dict of named tensors."""

    def __init__(self, model: torch.nn.Module):
        self.names = [n for n, _ in model.named_parameters()]
        self.shapes = [p.shape for _, p in model.named_parameters()]
        self.numels = [p.numel() for _, p in model.named_parameters()]
        self.d = sum(self.numels)

    def flatten(self, model: torch.nn.Module) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def to_dict(self, flat: torch.Tensor) -> dict[str, torch.Tensor]:
        """flat: [d] or [K, d] -> dict of [*shape] or [K, *shape]."""
        lead = flat.shape[:-1]
        parts = torch.split(flat, self.numels, dim=-1)
        return {n: p.reshape(*lead, *s) for n, p, s in zip(self.names, parts, self.shapes)}

    def from_dict(self, params: dict[str, torch.Tensor], k: int) -> torch.Tensor:
        return torch.cat([params[n].reshape(k, -1) for n in self.names], dim=1)


# --------------------------------------------------------------------------- local training
class VmapTrainer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.layout = ParamLayout(model)

        def loss_fn(params, x, y):
            out = functional_call(model, params, (x,))
            return F.cross_entropy(out, y)

        self._grad = vmap(grad(loss_fn))

    def local_updates(self, gvec, X, Y, epochs, batch, lr, perms=None, gen=None) -> torch.Tensor:
        """Train K clients from the global weights in parallel.

        X: [K, n, C, H, W], Y: [K, n]. perms: optional [epochs, K, n] data orders.
        Returns the K client updates (w_local - w_global) as a [K, d] matrix.
        """
        K, n = Y.shape
        flat = gvec.unsqueeze(0).expand(K, -1).clone()
        params = self.layout.to_dict(flat)
        rows = torch.arange(K, device=Y.device)[:, None]
        for e in range(epochs):
            if perms is not None:
                p = perms[e]
            else:
                p = torch.argsort(torch.rand(K, n, device=Y.device, generator=gen), dim=1)
            for i in range(0, n, batch):
                idx = p[:, i : i + batch]
                g = self._grad(params, X[rows, idx], Y[rows, idx])
                params = {k: params[k] - lr * g[k] for k in params}
        return self.layout.from_dict(params, K) - gvec


def local_update_sequential(model, layout: ParamLayout, gvec, X, Y, epochs, batch, lr, perms) -> torch.Tensor:
    """The notebook's per-client loop, kept for equivalence tests and benchmarks.
    X: [n, ...] for one client, perms: [epochs, n]."""
    torch.nn.utils.vector_to_parameters(gvec.clone(), model.parameters())
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    n = len(X)
    for e in range(epochs):
        for i in range(0, n, batch):
            idx = perms[e][i : i + batch]
            opt.zero_grad()
            F.cross_entropy(model(X[idx]), Y[idx]).backward()
            opt.step()
    return layout.flatten(model) - gvec


# --------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, gvec, X, Y, bs: int = 2000) -> dict[str, np.ndarray]:
    """Per-example loss, logit-scaled true-class confidence (phi) and correctness."""
    torch.nn.utils.vector_to_parameters(gvec.clone(), model.parameters())
    model.eval()
    loss, phi, correct = [], [], []
    for i in range(0, len(X), bs):
        z = model(X[i : i + bs]).float()
        y = Y[i : i + bs]
        zy = z.gather(1, y[:, None]).squeeze(1)
        others = z.scatter(1, y[:, None], float("-inf"))
        loss.append(F.cross_entropy(z, y, reduction="none"))
        phi.append(zy - torch.logsumexp(others, 1))  # log p_y - log(1 - p_y)
        correct.append(z.argmax(1) == y)
    cat = lambda xs: torch.cat(xs).cpu().numpy()
    return dict(loss=cat(loss), phi=cat(phi), correct=cat(correct))


# --------------------------------------------------------------------------- training loop
@dataclass
class RunResult:
    cfg: FLConfig
    seed: int
    evals: dict[str, dict[str, np.ndarray]]
    history: list[dict] = field(default_factory=list)
    diag: dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: dict[str, np.ndarray] = field(default_factory=dict)
    canary: CanaryResult | None = None
    seconds: float = 0.0


def run_fl(
    cfg: FLConfig,
    members: tuple[torch.Tensor, torch.Tensor],
    evals: dict[str, tuple[torch.Tensor, torch.Tensor]],
    seed: int,
    device,
    canaries: CanarySet | None = None,
    snapshot_n: int = 0,
    chunk: int = 128,
    log=print,
) -> RunResult:
    """Train one DP-FedAvg model.

    members: the training records, client k owns rows [k*S, (k+1)*S).
    evals: named (X, Y) sets evaluated on the final model, e.g. member/nonmember/test.
    snapshot_n: if > 0, per-example losses of the first snapshot_n members and
        non-members are stored at every eval point (used for the animations).
    """
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    gen = torch.Generator(device=device).manual_seed(seed)

    N, S = cfg.n_clients, cfg.samples_each
    Xm, Ym = members
    cX = Xm.view(N, S, *Xm.shape[1:])
    cY = Ym.view(N, S)

    model = CNN().to(device)
    trainer = VmapTrainer(model)
    g = trainer.layout.flatten(model).clone()
    g0 = g.clone()
    d = trainer.layout.d
    denom = cfg.q * N

    diag = {k: [] for k in ["n_sel", "norm_mean", "norm_med", "norm_max", "clip_frac", "signal", "noise", "cos_clean"]}
    history, snaps_m, snaps_n = [], [], []
    canary_dirs = canaries.all if canaries is not None else None
    n_ins = len(canaries.inserted) if canaries is not None else 0
    iter_max = None if canaries is None else torch.full((len(canary_dirs),), -1.0, device=device)

    def checkpoint(r):
        ev_t = evaluate(model, g, *evals["test"])
        row = dict(round=r, test_acc=float(ev_t["correct"].mean()))
        if "member" in evals and "nonmember" in evals:
            em, en = evaluate(model, g, *evals["member"]), evaluate(model, g, *evals["nonmember"])
            y = np.r_[np.ones(len(em["loss"])), np.zeros(len(en["loss"]))]
            row["train_acc"] = float(em["correct"].mean())
            row["auc"] = float(roc_auc_score(y, -np.r_[em["loss"], en["loss"]]))
            if snapshot_n:
                snaps_m.append(em["loss"][:snapshot_n])
                snaps_n.append(en["loss"][:snapshot_n])
        history.append(row)
        return row

    t0 = time.time()
    if snapshot_n:
        checkpoint(0)
    for r in range(cfg.rounds):
        sel = np.where(rng.rand(N) < cfg.q)[0]
        sel_t = torch.as_tensor(sel, device=device)
        deltas = torch.cat(
            [
                trainer.local_updates(g, cX[sel_t[i : i + chunk]], cY[sel_t[i : i + chunk]],
                                      cfg.local_epochs, cfg.batch, cfg.local_lr, gen=gen)
                for i in range(0, len(sel), chunk)
            ]
        ) if len(sel) else torch.zeros(0, d, device=device)

        norms = deltas.norm(dim=1)
        if cfg.clip is not None:
            deltas = deltas * torch.clamp(cfg.clip / (norms + 1e-12), max=1.0)[:, None]
        clean = deltas.sum(0)
        agg = clean.clone()
        if canaries is not None:
            part = torch.as_tensor(rng.rand(n_ins) < cfg.q, device=device)
            agg += (cfg.clip or 1.0) * canaries.inserted[part].sum(0)
        noise_norm = 0.0
        if cfg.is_dp:
            noise = torch.randn(d, device=device, generator=gen) * cfg.z * cfg.clip
            noise_norm = float(noise.norm())
            agg += noise
        if canaries is not None:
            cos = canary_dirs @ agg / (agg.norm() + 1e-12)
            iter_max = torch.maximum(iter_max, cos)
        g = g + cfg.server_lr * agg / denom

        diag["n_sel"].append(len(sel))
        diag["norm_mean"].append(float(norms.mean()) if len(sel) else 0.0)
        diag["norm_med"].append(float(norms.median()) if len(sel) else 0.0)
        diag["norm_max"].append(float(norms.max()) if len(sel) else 0.0)
        diag["clip_frac"].append(float((norms > cfg.clip).float().mean()) if cfg.clip and len(sel) else 0.0)
        diag["signal"].append(float(clean.norm()))
        diag["noise"].append(noise_norm)
        diag["cos_clean"].append(float(torch.dot(agg, clean) / (agg.norm() * clean.norm() + 1e-12)))

        if (r + 1) % cfg.eval_every == 0 or r == cfg.rounds - 1:
            row = checkpoint(r + 1)
            if log:
                extra = f" auc={row['auc']:.3f}" if "auc" in row else ""
                log(f"  [{cfg.name} s{seed}] round {r + 1}/{cfg.rounds} test_acc={row['test_acc']:.3f}{extra} ({time.time() - t0:.0f}s)")

    res = RunResult(cfg=cfg, seed=seed, evals={k: evaluate(model, g, *v) for k, v in evals.items()})
    res.history = history
    res.diag = {k: np.asarray(v) for k, v in diag.items()}
    if snapshot_n:
        res.snapshots = dict(member=np.stack(snaps_m), nonmember=np.stack(snaps_n),
                             rounds=np.array([h["round"] for h in history]))
    if canaries is not None:
        final = canary_dirs @ (g - g0) / ((g - g0).norm() + 1e-12)
        f, it = final.cpu().numpy(), iter_max.cpu().numpy()
        res.canary = CanaryResult(final_obs=f[:n_ins], final_null=f[n_ins:], iter_obs=it[:n_ins], iter_null=it[n_ins:])
    res.seconds = time.time() - t0
    return res
