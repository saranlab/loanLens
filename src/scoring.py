"""Score an applicant from the exported scorecard. Standard library only.

This module is the whole model at serving time, and it deliberately imports
nothing beyond the standard library. No numpy, no pandas, no sklearn. That is
what lets the API container skip the training stack entirely, and it is enforced
by a test rather than left as an intention, because an import added here without
thinking would quietly triple the image and couple the service to library
versions it has no reason to care about.

Kept separate from `serving.py`, which builds the JSON and does need pandas.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class ScoringModel:
    """Score applicants from the exported JSON.

    Small on purpose: this is what runs in front of applicants, and whoever has
    to sign off on it should be able to read the whole thing.
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
            # Right-closed, matching pd.cut on the training side.
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
            [float(b["points"]) for b in spec["bins"]]
            + [float(spec["missing"]["points"])]
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
