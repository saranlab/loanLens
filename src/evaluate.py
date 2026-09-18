"""Discrimination and calibration metrics for a credit model.

Accuracy is deliberately absent. At a 6.68% bad rate, predicting "good" for
everyone scores 93.3%, so the number carries no information about the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve


def ks_statistic(y_true, y_score) -> float:
    """Largest gap between the cumulative good and cumulative bad distributions.

    Computed off the ROC curve, where tpr is the cumulative share of bads and fpr
    the cumulative share of goods, so max(tpr - fpr) is the same quantity.

    `y_score` must rank bads high. Pass a probability of default, not a credit
    score, or invert the score first.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def ks_table(y_true, score, bins: int = 10) -> pd.DataFrame:
    """Decile table over the credit score: the shape a credit team expects.

    `score` is a credit score, so higher is safer and the table runs from the
    worst decile to the best.
    """
    y_true = np.asarray(y_true)
    df = pd.DataFrame({"score": np.asarray(score, dtype=float), "bad": y_true})
    df["decile"] = pd.qcut(df["score"], bins, labels=False, duplicates="drop")

    out = df.groupby("decile").agg(
        n=("bad", "size"),
        bad=("bad", "sum"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    )
    out["good"] = out["n"] - out["bad"]
    out["bad_rate"] = out["bad"] / out["n"]
    out["cum_bad_pct"] = out["bad"].cumsum() / out["bad"].sum()
    out["cum_good_pct"] = out["good"].cumsum() / out["good"].sum()
    out["ks"] = (out["cum_bad_pct"] - out["cum_good_pct"]).abs()
    return out


def calibration_table(y_true, p_bad, bins: int = 10) -> pd.DataFrame:
    """Predicted default rate against observed, per predicted-risk bucket.

    Discrimination and calibration are different questions. A model can rank
    perfectly and still be wrong about the level, which matters as soon as the
    output is used as a PD for pricing or provisioning rather than just for a
    yes/no decision.
    """
    df = pd.DataFrame({"p": np.asarray(p_bad, dtype=float), "bad": np.asarray(y_true)})
    df["bucket"] = pd.qcut(df["p"], bins, labels=False, duplicates="drop")
    out = df.groupby("bucket").agg(
        n=("bad", "size"), predicted=("p", "mean"), observed=("bad", "mean")
    )
    out["gap"] = out["observed"] - out["predicted"]
    return out


def metrics(y_true, p_bad) -> dict[str, float]:
    """Headline numbers. `p_bad` is a probability of default, so higher is worse."""
    auc = float(roc_auc_score(y_true, p_bad))
    return {
        "auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, p_bad),
        "pr_auc": float(average_precision_score(y_true, p_bad)),
        "brier": float(np.mean((np.asarray(p_bad) - np.asarray(y_true)) ** 2)),
        "bad_rate": float(np.mean(y_true)),
        "mean_predicted": float(np.mean(p_bad)),
    }


def format_metrics(name: str, values: dict[str, float]) -> str:
    return (
        f"{name:<8} AUC {values['auc']:.4f}  Gini {values['gini']:.4f}  "
        f"KS {values['ks']:.4f}  PR-AUC {values['pr_auc']:.4f}  "
        f"Brier {values['brier']:.5f}"
    )
