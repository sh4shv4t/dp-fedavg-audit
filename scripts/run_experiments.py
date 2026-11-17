"""Run every experiment in the repo. Finished runs are skipped, so it is safe to resume.

    python scripts/run_experiments.py main      # 6 configs x 3 seeds (the notebook's grid)
    python scripts/run_experiments.py refs      # reference (OUT) models for LiRA
    python scripts/run_experiments.py canary    # client-level canary audit
    python scripts/run_experiments.py bug       # canary audit vs a deliberately buggy DP implementation
    python scripts/run_experiments.py scaling   # same epsilon, more clients per round
    python scripts/run_experiments.py all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dpfl_audit.canary import CanarySet  # noqa: E402
from dpfl_audit.config import PAPER_CONFIGS  # noqa: E402
from dpfl_audit.data import make_split  # noqa: E402
from dpfl_audit.fl import ParamLayout, run_fl  # noqa: E402
from dpfl_audit.io import exists, run_path, save_run  # noqa: E402
from dpfl_audit.model import CNN  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCALING_CLIENTS = [500, 1000, 2500, 5000]
SCALING_CONFIGS = ["FedAvg", "z=1", "z=2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["main", "refs", "canary", "bug", "scaling", "all"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-refs", type=int, default=4)
    ap.add_argument("--canaries", type=int, default=200)
    ap.add_argument("--held-out", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="10 rounds, for smoke tests")
    args = ap.parse_args()

    print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else "")
    split = make_split()
    mem = split.take(split.member_idx, DEV)
    evals = dict(
        member=mem,
        nonmember=split.take(split.nonmember_idx, DEV),
        test=split.take(split.test_idx, DEV),
    )
    fix = (lambda c: c.with_(rounds=10, eval_every=5)) if args.quick else (lambda c: c)
    todo = ["main", "refs", "canary", "bug", "scaling"] if args.what == "all" else [args.what]

    if "main" in todo:
        for seed in args.seeds:
            for cfg in PAPER_CONFIGS.values():
                p = run_path("main", cfg.name, seed)
                if exists(p):
                    continue
                print(f"=== main | {cfg.name} | seed {seed} ===")
                res = run_fl(fix(cfg), mem, evals, seed, DEV, snapshot_n=2000 if seed == 0 else 0)
                save_run(res, p)

    if "refs" in todo:
        # OUT models: same recipe, trained on a fresh 10k sample of the unused pool
        for r in range(args.n_refs):
            idx = np.random.RandomState(1000 + r).choice(split.pool_idx, 10_000, replace=False)
            ref_mem = split.take(idx, DEV)
            for cfg in PAPER_CONFIGS.values():
                p = run_path("refs", cfg.name, r)
                if exists(p):
                    continue
                print(f"=== ref {r} | {cfg.name} ===")
                res = run_fl(fix(cfg), ref_mem, evals, 1000 + r, DEV)
                save_run(res, p)
            del ref_mem

    if "canary" in todo:
        d = ParamLayout(CNN()).d
        for seed in args.seeds:
            cs = CanarySet.make(d, args.canaries, args.held_out, seed, DEV)
            for cfg in PAPER_CONFIGS.values():
                p = run_path("canary", cfg.name, seed)
                if exists(p):
                    continue
                print(f"=== canary | {cfg.name} | seed {seed} ===")
                res = run_fl(fix(cfg), mem, dict(test=evals["test"]), seed, DEV, canaries=cs)
                save_run(res, p)
            del cs

    if "bug" in todo:
        # an implementation bug the accountant cannot see; can the canary audit catch it?
        d = ParamLayout(CNN()).d
        cfg = PAPER_CONFIGS["z=2"].with_(name="z=2 buggy", noise_on_mean_bug=True)
        for seed in args.seeds:
            p = run_path("canary", cfg.name, seed)
            if exists(p):
                continue
            print(f"=== canary | {cfg.name} | seed {seed} ===")
            cs = CanarySet.make(d, args.canaries, args.held_out, seed, DEV)
            res = run_fl(fix(cfg), mem, dict(test=evals["test"]), seed, DEV, canaries=cs)
            save_run(res, p)
            del cs

    if "scaling" in todo:
        # total data fixed at 10k records, so more clients means fewer records each
        for n in SCALING_CLIENTS:
            for name in SCALING_CONFIGS:
                cfg = PAPER_CONFIGS[name].with_(n_clients=n, samples_each=10_000 // n)
                p = run_path("scaling", f"{name}_N{n}", 0)
                if exists(p):
                    continue
                print(f"=== scaling | {name} | N={n} ({cfg.q * n:.0f} clients/round) ===")
                res = run_fl(fix(cfg), mem, evals, 0, DEV)
                save_run(res, p)


if __name__ == "__main__":
    main()
