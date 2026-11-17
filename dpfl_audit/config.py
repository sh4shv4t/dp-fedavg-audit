from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class FLConfig:
    """One DP-FedAvg training setup. Defaults match the original notebook."""

    name: str = "FedAvg"
    n_clients: int = 500
    samples_each: int = 20
    q: float = 0.1  # Poisson client sampling rate per round
    rounds: int = 100
    local_epochs: int = 5
    batch: int = 10
    local_lr: float = 0.05
    server_lr: float = 1.0
    clip: float | None = None  # None means plain FedAvg (no clipping)
    z: float = 0.0  # noise multiplier, std of noise on the SUM is z * clip
    delta: float = 1e-5
    eval_every: int = 10

    @property
    def n_members(self) -> int:
        return self.n_clients * self.samples_each

    @property
    def is_dp(self) -> bool:
        return self.clip is not None and self.z > 0

    def with_(self, **kw) -> "FLConfig":
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


_BASE = FLConfig()
PAPER_CONFIGS: dict[str, FLConfig] = {
    "FedAvg": _BASE.with_(name="FedAvg", clip=None, z=0.0),
    "Clip-only": _BASE.with_(name="Clip-only", clip=1.0, z=0.0),
    "z=0.5": _BASE.with_(name="z=0.5", clip=1.0, z=0.5),
    "z=1": _BASE.with_(name="z=1", clip=1.0, z=1.0),
    "z=2": _BASE.with_(name="z=2", clip=1.0, z=2.0),
    "z=4": _BASE.with_(name="z=4", clip=1.0, z=4.0),
}
