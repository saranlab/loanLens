"""Export the scorecard as plain JSON, and score from it without sklearn.

A pickled estimator is a bad thing to hand a service. Unpickling needs the
training package importable, at compatible versions of sklearn, pandas and numpy,
so the scoring container ends up carrying the whole training stack and a library
upgrade can stop the model loading at all.

A scorecard does not need any of that. Once it is fit, the entire model is a
lookup: which bin does this value fall into, how many points does that bin carry.
That fits in a JSON file, which is also something a risk reviewer can read.

The cost of the split is that binning now exists twice, once in `binning.py` for
training and once in `src/scoring.py` for serving, and two implementations can
drift. `tests/test_serving.py` closes that by asserting the JSON scorer
reproduces the fitted `Scorecard` exactly across the whole test set.

This module builds the JSON and needs pandas. `src/scoring.py` reads it and needs
nothing outside the standard library, which is why the API container copies only
that one.

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
from .scoring import SCHEMA_VERSION, ScoringModel

# Re-exported so callers can import either name, but the service imports it from
# src.scoring directly, which is the module with no third-party dependencies.
__all__ = ["SCHEMA_VERSION", "ScoringModel", "export", "to_dict", "frame_to_applicants"]

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
