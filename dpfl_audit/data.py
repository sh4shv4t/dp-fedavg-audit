"""CIFAR-10 loading and the fixed member / non-member / test / reference split.

The split matches the notebook exactly: train and test sets are concatenated,
normalised with global channel statistics and permuted with RandomState(0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


@dataclass
class Split:
    X: torch.Tensor  # all 60k images, normalised, on CPU
    Y: torch.Tensor
    mean: torch.Tensor
    std: torch.Tensor
    member_idx: np.ndarray
    nonmember_idx: np.ndarray
    test_idx: np.ndarray
    pool_idx: np.ndarray  # never used by the target model; reference models train here

    def take(self, idx, device) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.as_tensor(idx)
        return self.X[idx].to(device), self.Y[idx].to(device)

    def denormalise(self, x: torch.Tensor) -> torch.Tensor:
        return (x.cpu() * self.std[0] + self.mean[0]).clamp(0, 1)


HF_URL = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{}-00000-of-00001.parquet"


def _from_parquet(root: Path) -> tuple[np.ndarray, np.ndarray]:
    """CIFAR-10 from the Hugging Face mirror (same images and order as the original
    batches). The official Toronto server can be very slow, so this is the default."""
    import io
    import urllib.request

    import pandas as pd
    from PIL import Image

    xs, ys = [], []
    for part in ["train", "test"]:
        f = root / "hf" / f"{part}.parquet"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HF_URL.format(part), f)
        df = pd.read_parquet(f)
        xs.append(np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["img"]]))
        ys.append(df["label"].to_numpy())
    return np.concatenate(xs), np.concatenate(ys)


def load_cifar10(root: str = "./data") -> tuple[torch.Tensor, torch.Tensor]:
    """All 60k CIFAR-10 images (train then test) as float tensors in [0, 1]."""
    root = Path(root)
    cache = root / "cifar10_uint8.npz"
    if not cache.exists():
        if (root / "cifar-10-batches-py").exists():  # torchvision copy, if you already have it
            import torchvision

            tr = torchvision.datasets.CIFAR10(str(root), train=True)
            te = torchvision.datasets.CIFAR10(str(root), train=False)
            X, Y = np.concatenate([tr.data, te.data]), np.concatenate([tr.targets, te.targets])
        else:
            X, Y = _from_parquet(root)
        np.savez(cache, X=X, Y=Y)
    with np.load(cache) as z:
        X, Y = z["X"], z["Y"]
    return torch.tensor(X).permute(0, 3, 1, 2).float() / 255.0, torch.tensor(Y).long()


def make_split(n_members: int = 10_000, root: str = "./data") -> Split:
    X, Y = load_cifar10(root)
    mean = X.mean((0, 2, 3), keepdim=True)
    std = X.std((0, 2, 3), keepdim=True)
    X = (X - mean) / std
    perm = np.random.RandomState(0).permutation(len(X))
    m = n_members
    return Split(
        X=X,
        Y=Y,
        mean=mean,
        std=std,
        member_idx=perm[:m],
        nonmember_idx=perm[m : 2 * m],
        test_idx=perm[2 * m : 3 * m],
        pool_idx=perm[3 * m :],
    )
