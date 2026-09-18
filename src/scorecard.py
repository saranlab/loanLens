"""Turn a logistic regression on WoE features into a points scorecard.

The sign bookkeeping here is the part worth reading twice.

`LogisticRegression` predicts P(bad), so its linear term is

    logit = b0 + sum(b_i * woe_i) = ln(p_bad / p_good)

while a credit score runs the other way: more points means safer. Writing the
odds the scorecard cares about as odds = p_good / p_bad gives ln(odds) = -logit,
so every conversion below carries a minus sign. Get it wrong and the scorecard
still produces plausible numbers, ranked backwards, with nothing raising an
error. `test_scorecard.py` pins the direction.

Scaling follows the usual convention: a chosen score at chosen odds, and a fixed
number of points to double the odds.

    factor = PDO / ln(2)
    offset = base_score - factor * ln(base_odds)
    score  = offset + factor * ln(odds)

With PDO 20 and 600 points at 50:1, that reads as 600 -> 50:1, 620 -> 100:1,
640 -> 200:1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config as cfg
from .binning import MISSING, bin_labels
from .binning import WoEBinner


def scaling_constants(
    base_score: float = cfg.BASE_SCORE,
    base_odds: float = cfg.BASE_ODDS,
    pdo: float = cfg.PDO,
) -> tuple[float, float]:
    """Return (factor, offset) for the score scale."""
    factor = pdo / np.log(2.0)
    offset = base_score - factor * np.log(base_odds)
    return float(factor), float(offset)


def odds_to_score(odds, *, base_score=cfg.BASE_SCORE, base_odds=cfg.BASE_ODDS, pdo=cfg.PDO):
    """Good:bad odds to points."""
    factor, offset = scaling_constants(base_score, base_odds, pdo)
    return offset + factor * np.log(odds)


def score_to_odds(score, *, base_score=cfg.BASE_SCORE, base_odds=cfg.BASE_ODDS, pdo=cfg.PDO):
    """Points back to good:bad odds. The inverse of odds_to_score."""
    factor, offset = scaling_constants(base_score, base_odds, pdo)
    return np.exp((np.asarray(score, dtype=float) - offset) / factor)


def probability_to_score(p_bad, **kw):
    """P(bad) to points, guarding the two ends so the log stays finite."""
    p = np.clip(np.asarray(p_bad, dtype=float), 1e-12, 1 - 1e-12)
    return odds_to_score((1.0 - p) / p, **kw)


def tier_cutoffs(max_pd=None, **kw) -> list[tuple[float, str]]:
    """Convert each tier's maximum default probability into a score cutoff.

    Derived rather than hardcoded, so changing the scale moves the cutoffs with
    it instead of leaving them stranded at numbers from a different scale.
    """
    pairs = max_pd if max_pd is not None else cfg.TIER_MAX_PD
    return [(float(probability_to_score(pd_max, **kw)), name) for pd_max, name in pairs]


def tier_for(score, cutoffs=None, floor: str = cfg.TIER_FLOOR, **kw) -> str:
    """Map points to a risk tier. Cutoffs are read highest first."""
    for threshold, name in cutoffs if cutoffs is not None else tier_cutoffs(**kw):
        if score >= threshold:
            return name
    return floor


class Scorecard:
    """A fitted binner plus a fitted logistic model, exposed as points.

    Built from components that were already fit, so this class adds no fitting of
    its own and cannot introduce leakage.
    """

    def __init__(
        self,
        binner: WoEBinner,
        model: LogisticRegression,
        *,
        base_score: float = cfg.BASE_SCORE,
        base_odds: float = cfg.BASE_ODDS,
        pdo: float = cfg.PDO,
    ):
        self.binner = binner
        self.model = model
        self.base_score = base_score
        self.base_odds = base_odds
        self.pdo = pdo

        self.factor, self.offset = scaling_constants(base_score, base_odds, pdo)
        self.features = list(binner.features_)

        coefs = np.asarray(model.coef_).ravel()
        if len(coefs) != len(self.features):
            raise ValueError(
                f"model has {len(coefs)} coefficients but binner kept "
                f"{len(self.features)} features"
            )
        self.coef_ = dict(zip(self.features, coefs))
        self.intercept_ = float(np.asarray(model.intercept_).ravel()[0])

        # The whole score at logit 0, before any feature contributes.
        self.base_points = self.offset - self.factor * self.intercept_

    def points_for_bin(self, feature: str, woe: float) -> float:
        """Points a single bin contributes. Minus sign: see the module docstring."""
        return -self.factor * self.coef_[feature] * woe

    def points_table(self) -> pd.DataFrame:
        """The scorecard itself: one row per bin, with the points it carries.

        This table is what gets handed to a credit officer, and what an adverse
        action notice is written from.
        """
        rows = []
        for feature in self.features:
            table = self.binner.tables_[feature]
            for bin_label, row in table.iterrows():
                rows.append(
                    {
                        "feature": feature,
                        "bin": bin_label,
                        "n_train": int(row["n"]),
                        "share": float(row["share"]),
                        "bad_rate": float(row["bad_rate"]),
                        "woe": float(row["woe"]),
                        "points": self.points_for_bin(feature, float(row["woe"])),
                    }
                )
        out = pd.DataFrame(rows)
        out["points"] = out["points"].round(2)
        return out

    def predict_proba_bad(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(self.binner.transform(X))[:, 1]

    def score(self, X: pd.DataFrame) -> pd.Series:
        """Credit score per row, computed through the probability."""
        p = self.predict_proba_bad(X)
        return pd.Series(probability_to_score(p, base_score=self.base_score,
                                              base_odds=self.base_odds, pdo=self.pdo),
                         index=X.index, name="score")

    def score_from_points(self, X: pd.DataFrame) -> pd.Series:
        """Same score, summed from the points table instead.

        Exists so the two routes can be checked against each other. If they
        disagree, the points table a human reads does not match the model that
        made the decision, which is the failure an audit is looking for.
        """
        woe = self.binner.transform(X)
        total = np.full(len(X), self.base_points, dtype=float)
        for feature in self.features:
            total += -self.factor * self.coef_[feature] * woe[f"woe__{feature}"].to_numpy()
        return pd.Series(total, index=X.index, name="score")

    def explain(self, X: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        """Per applicant, the features costing the most points against their best.

        Adverse action reasons: a declined applicant is owed the reasons, and the
        honest answer is which bins cost them the most relative to the best bin
        available on that feature.
        """
        woe = self.binner.transform(X)
        best = {
            f: max(self.points_for_bin(f, w) for w in self.binner.woe_[f].values())
            for f in self.features
        }
        shortfall = pd.DataFrame(
            {
                f: best[f] - (-self.factor * self.coef_[f] * woe[f"woe__{f}"])
                for f in self.features
            },
            index=X.index,
        )
        ranked = shortfall.apply(
            lambda row: [
                f"{name} (-{row[name]:.0f})"
                for name in row.sort_values(ascending=False).index[:top_n]
            ],
            axis=1,
        )
        return pd.DataFrame(
            {"score": self.score(X), "reasons": ranked}, index=X.index
        )

    def bin_for(self, feature: str, value) -> str:
        """Which bin a raw value falls into, for a single applicant."""
        label = bin_labels(pd.Series([value]), feature).iloc[0]
        return label if label in self.binner.woe_[feature] else MISSING
