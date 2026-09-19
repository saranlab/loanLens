"""Export the scorecard as plain JSON, and score from it without sklearn.

A pickled estimator is a bad thing to hand a service. Unpickling needs the
training package importable, at compatible versions of sklearn, pandas and numpy,
so the scoring container ends up carrying the whole training stack and a library
upgrade can stop the model loading at all.

A scorecard does not need any of that. Once it is fit, the entire model is a
lookup: which bin does this value fall into, how many points does that bin carry.
That fits in a JSON file, which is also something a risk reviewer can read.

The cost of the split is that binning now exists twice, once in `binning.py` for
training and once in `ScoringModel` for serving, and two implementations can
drift. `tests/test_serving.py` closes that by asserting the JSON scorer
reproduces the fitted `Scorecard` exactly across the whole test set.

To keep the two from drifting in the first place, the JSON stores numeric bin
boundaries rather than the label strings pandas happens to produce, so nothing
here depends on how `pd.cut` formats an interval.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config as cfg
from .binning import MISSING, bin_labels
from .scorecard import tier_cutoffs

SCHEMA_VERSION = 1

# JSON has no Infinity, so an unbounded edge is written as null.
def _edge(value: float) -> float | None:
    return None if math.isinf(value) else float(value)


def _interval_bins(card, feature: str) -> list[dict[str, Any]]:
    """Ordered bins for a feature cut by fixed edges, right-closed.

    Labels come from `bin_labels` on each interval's midpoint rather than from
    reconstructing the string pandas would produce, so the mapping is correct by
    construction rather than by matching formats.
    """
    edges = cfg.BIN_EDGES[feature]
    woe_map = card.binner.woe_[feature]
    bins = []
    for left, right in zip(edges, edges[1:]):
        if math.isinf(left) and math.isinf(right):
            raise ValueError(f"{feature}: a single unbounded bin is not a binning")
        # a point strictly inside the interval
        if math.isinf(left):
            probe = right - 1.0
        elif math.isinf(right):
            probe = left + 1.0
        else:
            probe = (left + right) / 2.0

        label = bin_labels(pd.Series([probe]), feature).iloc[0]
        if label not in woe_map:
            raise ValueError(f"{feature}: probe {probe} produced unknown bin {label}")

        woe = float(woe_map[label])
        bins.append(
            {
                "left": _edge(left),
                "right": _edge(right),
                "woe": woe,
                "points": card.points_for_bin(feature, woe),
            }
        )
    return bins


def _value_bins(card, feature: str) -> list[dict[str, Any]]:
    """Bins for a discrete feature, one per observed value after clipping."""
    woe_map = card.binner.woe_[feature]
    bins = []
    for label, woe in woe_map.items():
        if label == MISSING:
            continue
        bins.append(
            {
                "value": float(label),
                "woe": float(woe),
                "points": card.points_for_bin(feature, float(woe)),
            }
        )
    return sorted(bins, key=lambda b: b["value"])


def to_dict(card, metrics: dict | None = None) -> dict[str, Any]:
    """Everything needed to score one applicant, as plain data."""
    features = []
    for name in card.features:
        woe_map = card.binner.woe_[name]
        missing_woe = float(woe_map.get(MISSING, 0.0))
        entry: dict[str, Any] = {
            "name": name,
            "coef": float(card.coef_[name]),
            "iv": float(card.binner.iv_[name]),
            "missing": {
                "woe": missing_woe,
                "points": card.points_for_bin(name, missing_woe),
            },
        }
        if name in cfg.BIN_EDGES:
            entry["kind"] = "interval"
            entry["bins"] = _interval_bins(card, name)
        else:
            entry["kind"] = "value"
            entry["clip_at"] = cfg.CLIP_AT.get(name)
            entry["bins"] = _value_bins(card, name)
        features.append(entry)

    return {
        "schema_version": SCHEMA_VERSION,
        "scaling": {
            "base_score": card.base_score,
            "base_odds": card.base_odds,
            "pdo": card.pdo,
            "factor": card.factor,
            "offset": card.offset,
        },
        "base_points": card.base_points,
        "intercept": card.intercept_,
        "features": features,
        "tiers": [
            {"min_score": float(pts), "name": name} for pts, name in tier_cutoffs()
        ],
        "tier_floor": cfg.TIER_FLOOR,
        "metrics": metrics or {},
    }


def export(card, path: Path, metrics: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(card, metrics), indent=2), encoding="utf-8")
    return path


class ScoringModel:
    """Score applicants from the exported JSON. No sklearn, no pandas needed.

    Deliberately small and dependency-light: this is what the API container runs,
    and it should be readable end to end by whoever has to sign off on it.
    """

    def __init__(self, spec: dict[str, Any]):
        if spec.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {spec.get('schema_version')}, "
                f"this code reads {SCHEMA_VERSION}"
            )
        self.spec = spec
        self.scaling = spec["scaling"]
        self.base_points = float(spec["base_points"])
        self.features = {f["name"]: f for f in spec["features"]}
        self.feature_names = [f["name"] for f in spec["features"]]
        self.tiers = spec["tiers"]
        self.tier_floor = spec["tier_floor"]

    @classmethod
    def from_json(cls, path: str | Path) -> "ScoringModel":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    @staticmethod
    def _is_missing(value) -> bool:
        return value is None or (isinstance(value, float) and math.isnan(value))

    def points_for(self, name: str, value) -> float:
        """Points one feature contributes for one applicant."""
        spec = self.features[name]
        if self._is_missing(value):
            return float(spec["missing"]["points"])

        value = float(value)
        if spec["kind"] == "interval":
            for b in spec["bins"]:
                above = b["left"] is None or value > b["left"]
                below = b["right"] is None or value <= b["right"]
                if above and below:
                    return float(b["points"])
            # unreachable while the outer edges stay unbounded
            return 0.0

        clip_at = spec.get("clip_at")
        if clip_at is not None:
            value = min(value, float(clip_at))
        for b in spec["bins"]:
            if math.isclose(b["value"], value, rel_tol=0.0, abs_tol=1e-9):
                return float(b["points"])
        # A value never seen in training scores at the portfolio average, which is
        # what WoEBinner.transform does with an unseen bin.
        return 0.0

    def breakdown(self, applicant: dict[str, Any]) -> dict[str, float]:
        """Points per feature, so a decision can be explained line by line."""
        return {
            name: self.points_for(name, applicant.get(name))
            for name in self.feature_names
        }

    def score(self, applicant: dict[str, Any]) -> float:
        return self.base_points + sum(self.breakdown(applicant).values())

    def probability(self, applicant: dict[str, Any]) -> float:
        """Back out P(default) from the score, inverting the PDO relation."""
        odds = math.exp(
            (self.score(applicant) - self.scaling["offset"]) / self.scaling["factor"]
        )
        return 1.0 / (1.0 + odds)

    def tier(self, score: float) -> str:
        for t in self.tiers:
            if score >= t["min_score"]:
                return t["name"]
        return self.tier_floor

    def best_points(self, name: str) -> float:
        spec = self.features[name]
        return max(
            [float(b["points"]) for b in spec["bins"]] + [float(spec["missing"]["points"])]
        )

    def reasons(self, applicant: dict[str, Any], top_n: int = 3) -> list[dict[str, Any]]:
        """Features costing the most points against the best bin available.

        These are adverse action reasons: what a declined applicant is owed.
        """
        gaps = [
            {
                "feature": name,
                "points": pts,
                "points_lost": self.best_points(name) - pts,
            }
            for name, pts in self.breakdown(applicant).items()
        ]
        gaps.sort(key=lambda g: g["points_lost"], reverse=True)
        return [g for g in gaps[:top_n] if g["points_lost"] > 0.005]

    def assess(self, applicant: dict[str, Any], top_n: int = 3) -> dict[str, Any]:
        score = self.score(applicant)
        return {
            "score": round(score, 1),
            "probability_of_default": round(self.probability(applicant), 5),
            "tier": self.tier(score),
            "base_points": round(self.base_points, 1),
            "points": {k: round(v, 2) for k, v in self.breakdown(applicant).items()},
            "reasons": [
                {
                    "feature": r["feature"],
                    "points": round(r["points"], 2),
                    "points_lost": round(r["points_lost"], 2),
                }
                for r in self.reasons(applicant, top_n)
            ],
        }


def frame_to_applicants(X: pd.DataFrame, names: list[str]) -> list[dict[str, Any]]:
    """DataFrame rows to plain dicts, with NaN turned into None.

    Used by the parity test to feed the same rows through both paths.
    """
    out = []
    for _, row in X[names].iterrows():
        out.append(
            {n: (None if pd.isna(row[n]) else float(row[n])) for n in names}
        )
    return out
