"""A transparent proximity heuristic, not a proof of causal overlap."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .schema import PLAN_COLUMNS, W_COLUMNS, numeric, patient_features


def joint_features(cases: pd.DataFrame, plans: np.ndarray) -> np.ndarray:
    return np.column_stack([patient_features(cases), numeric(cases, W_COLUMNS), plans])


@dataclass
class SupportGuard:
    scaler: StandardScaler
    neighbors: NearestNeighbors
    threshold: float
    minimum: np.ndarray
    maximum: np.ndarray
    k: int

    @classmethod
    def fit(cls, train: pd.DataFrame, k: int = 8) -> "SupportGuard":
        raw = joint_features(train, numeric(train, PLAN_COLUMNS))
        if len(raw) <= k:
            raise ValueError("Support guard needs more rows than neighbors.")
        scaler = StandardScaler().fit(raw)
        scaled = scaler.transform(raw)
        neighbors = NearestNeighbors().fit(scaled)
        # The k+1 query includes self; the last distance is the kth OTHER case.
        distances, _ = neighbors.kneighbors(scaled, n_neighbors=k + 1)
        threshold = float(np.quantile(distances[:, -1], 0.95))
        return cls(scaler, neighbors, threshold, raw.min(axis=0), raw.max(axis=0), k)

    def check(self, patient: pd.DataFrame, plans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if len(patient) != 1:
            raise ValueError("Evaluate one patient at a time.")
        cases = pd.concat([patient] * len(plans), ignore_index=True)
        raw = joint_features(cases, plans)
        distances, _ = self.neighbors.kneighbors(self.scaler.transform(raw), n_neighbors=self.k)
        distance = distances[:, -1]
        in_ranges = ((raw >= self.minimum) & (raw <= self.maximum)).all(axis=1)
        return in_ranges & (distance <= self.threshold), distance
