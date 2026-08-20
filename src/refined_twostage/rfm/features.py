"""Recovered sine random-feature generator."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SineFeatureBank:
    """Feature bank phi_j(x)=sin(alpha*(x*w_j+b_j))."""

    w: np.ndarray
    b: np.ndarray
    alpha: float
    seed: int
    distribution: str = "uniform_w_then_uniform_b"
    low: float = -1.0
    high: float = 1.0

    def phase(self, x):
        xx = np.asarray(x, dtype=float).reshape(-1)
        return self.alpha * (xx[:, None] * self.w[None, :] + self.b[None, :])

    def values(self, x):
        return np.sin(self.phase(x))

    def first_derivatives(self, x):
        return self.alpha * self.w[None, :] * np.cos(self.phase(x))

    def second_derivatives(self, x):
        return -((self.alpha * self.w[None, :]) ** 2) * np.sin(self.phase(x))

    def prefix(self, n_features: int):
        n = int(n_features)
        return SineFeatureBank(np.asarray(self.w[:n], dtype=float).copy(), np.asarray(self.b[:n], dtype=float).copy(), float(self.alpha), int(self.seed), self.distribution + "_prefix{}".format(n), float(self.low), float(self.high))


def sample_uniform_sine_features(n_features: int, seed: int = 7061015, alpha: float = 50.0) -> SineFeatureBank:
    rng = np.random.default_rng(int(seed))
    w = rng.uniform(-1.0, 1.0, int(n_features))
    b = rng.uniform(-1.0, 1.0, int(n_features))
    return SineFeatureBank(np.asarray(w, dtype=float), np.asarray(b, dtype=float), float(alpha), int(seed))
