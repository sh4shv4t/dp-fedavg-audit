"""Benchmark: the notebook's per-client loop vs vmap'd client training.

Times one FL round (K clients x 20 records x 5 local epochs) for several K and
writes results/benchmark.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpfl_audit.fl import VmapTrainer, local_update_sequential  # noqa: E402
from dpfl_audit.model import CNN  # noqa: E402

DEV = "cuda"
N_SAMPLES, EPOCHS, BATCH, LR = 20, 5, 10, 0.05


def timed(fn, reps):
    fn()  # warm-up
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def main():
    torch.manual_seed(0)
    model = CNN().to(DEV)
    tr = VmapTrainer(model)
    g = tr.layout.flatten(model).clone()
    seq_model = CNN().to(DEV)
    rows = []
    for K in [10, 50, 100, 250, 500]:
        X = torch.randn(K, N_SAMPLES, 3, 32, 32, device=DEV)
        Y = torch.randint(0, 10, (K, N_SAMPLES), device=DEV)
        perms = torch.stack([torch.argsort(torch.rand(K, N_SAMPLES, device=DEV), 1) for _ in range(EPOCHS)])

        def seq():
            for k in range(K):
                local_update_sequential(seq_model, tr.layout, g, X[k], Y[k], EPOCHS, BATCH, LR, perms[:, k])

        def vec():
            for i in range(0, K, 128):
                tr.local_updates(g, X[i : i + 128], Y[i : i + 128], EPOCHS, BATCH, LR, perms=perms[:, i : i + 128])

        reps = 3 if K <= 100 else 1
        ts, tv = timed(seq, reps), timed(vec, max(reps, 3))
        rows.append(dict(clients=K, sequential_s=ts, vmap_s=tv, speedup=ts / tv))
        print(f"K={K:4d}  sequential {ts * 1000:8.1f} ms  vmap {tv * 1000:7.1f} ms  speedup x{ts / tv:.1f}")

    out = dict(device=torch.cuda.get_device_name(0), torch=torch.__version__, rows=rows)
    Path("results").mkdir(exist_ok=True)
    Path("results/benchmark.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
