"""Saving and loading run results (compact .npz + .json pairs under results/raw)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .fl import RunResult

RAW = Path("results/raw")


def run_path(kind: str, name: str, seed: int) -> Path:
    safe = name.replace("=", "").replace(".", "p")
    return RAW / kind / f"{safe}_s{seed}"


def save_run(res: RunResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for split, ev in res.evals.items():
        if split == "test":
            continue
        arrays[f"{split}_loss"] = ev["loss"].astype(np.float32)
        arrays[f"{split}_phi"] = ev["phi"].astype(np.float32)
    for k, v in res.diag.items():
        arrays[f"diag_{k}"] = v
    for k, v in res.snapshots.items():
        arrays[f"snap_{k}"] = v.astype(np.float32) if v.dtype.kind == "f" else v
    if res.canary is not None:
        for k, v in vars(res.canary).items():
            arrays[f"canary_{k}"] = v
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    meta = dict(
        cfg=res.cfg.to_dict(),
        seed=res.seed,
        seconds=res.seconds,
        accuracy={k: float(v["correct"].mean()) for k, v in res.evals.items()},
        history=res.history,
    )
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def load_run(path: Path) -> tuple[dict, dict]:
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz")) as z:
        arrays = {k: z[k] for k in z.files}
    return meta, arrays


def exists(path: Path) -> bool:
    return path.with_suffix(".json").exists()
