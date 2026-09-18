"""Fit the scorecard end to end and write the artifact.

    python -m src.train

Ordering is the thing to read here. The split comes first, and every fitted
object after it sees training rows only. The test set is touched exactly once, by
transform, and only to be scored.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as cfg
from . import evaluate as ev
from .binning import WoEBinner
from .preprocessing import add_features, clean, load_raw
from .scorecard import Scorecard, tier_cutoffs, tier_for

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_CSV = ROOT / "data" / "raw" / "cs-training.csv"
DEFAULT_ARTIFACT = ROOT / "artifacts" / "scorecard.joblib"


def build_dataset(csv_path: Path):
    """Load, clean and derive features. None of this looks at the target."""
    raw = load_raw(csv_path)
    y = raw[cfg.TARGET].astype(int)
    X = add_features(clean(raw)).drop(columns=[cfg.TARGET])
    return X, y


def duplicate_groups(X: pd.DataFrame) -> np.ndarray:
    """Group id per row, identical for rows that are exact duplicates.

    Section 5.4 found 609 rows duplicated across every column. With no customer
    id there is no telling repeats from coincidence, so they stay in the data, but
    letting the same profile sit on both sides of the split makes the test score
    optimistic. Grouping them lets the split keep them together.
    """
    key = X.astype(str).agg("|".join, axis=1)
    return key.factorize()[0]


def split(X: pd.DataFrame, y: pd.Series):
    """Stratified split that keeps exact duplicates on one side.

    Stratifying preserves the 6.68% bad rate in both halves, which matters at
    this level of imbalance: a plain random split can move the test bad rate by
    enough to shift every metric.
    """
    groups = duplicate_groups(X)
    counts = pd.Series(groups).value_counts()
    shared = counts[counts > 1].index

    if len(shared) == 0:
        return train_test_split(
            X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE
        )

    # Split the unique rows normally, then assign each duplicate group whole.
    is_dup = pd.Series(groups, index=X.index).isin(shared)
    X_uniq, y_uniq = X[~is_dup], y[~is_dup]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_uniq, y_uniq, test_size=cfg.TEST_SIZE, stratify=y_uniq,
        random_state=cfg.RANDOM_STATE,
    )

    rng = np.random.default_rng(cfg.RANDOM_STATE)
    dup_index = X.index[is_dup]
    dup_groups = pd.Series(groups, index=X.index)[is_dup]
    to_test = {g for g in shared if rng.random() < cfg.TEST_SIZE}
    test_rows = dup_index[dup_groups.isin(to_test)]
    train_rows = dup_index.difference(test_rows)

    X_tr = pd.concat([X_tr, X.loc[train_rows]])
    y_tr = pd.concat([y_tr, y.loc[train_rows]])
    X_te = pd.concat([X_te, X.loc[test_rows]])
    y_te = pd.concat([y_te, y.loc[test_rows]])
    return X_tr, X_te, y_tr, y_te


def fit_scorecard(X_tr: pd.DataFrame, y_tr: pd.Series) -> tuple[Scorecard, WoEBinner]:
    binner = WoEBinner(cfg.MODEL_FEATURES).fit(X_tr, y_tr)
    woe_tr = binner.transform(X_tr)

    # No class weighting. At a 6.7% bad rate it is not needed for ranking, and it
    # would decalibrate the probabilities the score is derived from. L2 with a
    # large C because the features are already WoE encoded and few in number.
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=cfg.RANDOM_STATE)
    model.fit(woe_tr, y_tr)
    return Scorecard(binner, model), binner


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args(argv)

    if not args.csv.exists():
        parser.error(
            f"{args.csv} not found. Download the dataset from Kaggle first; see README."
        )

    X, y = build_dataset(args.csv)
    print(f"loaded {len(X):,} rows, bad rate {y.mean():.4%}")

    X_tr, X_te, y_tr, y_te = split(X, y)
    print(
        f"train {len(X_tr):,} ({y_tr.mean():.4%} bad)   "
        f"test {len(X_te):,} ({y_te.mean():.4%} bad)"
    )

    card, binner = fit_scorecard(X_tr, y_tr)

    print("\nIV and monotonicity, fit on train only")
    print(binner.iv_summary().round(4).to_string())
    if binner.dropped_:
        print("dropped below min_iv:", binner.dropped_)
    if binner.thin_bins_:
        print("bins pinned to the portfolio average:", binner.thin_bins_)

    p_tr = card.predict_proba_bad(X_tr)
    p_te = card.predict_proba_bad(X_te)
    m_tr, m_te = ev.metrics(y_tr, p_tr), ev.metrics(y_te, p_te)

    # Cross-validated on the training half, refitting the binner inside each fold,
    # so the fold's own rows never shape its WoE. A single holdout number says
    # little on its own; the spread across folds says whether it is stable.
    folds = StratifiedKFold(n_splits=args.cv_folds, shuffle=True,
                            random_state=cfg.RANDOM_STATE)
    cv_scores = []
    for fold_tr, fold_va in folds.split(X_tr, y_tr):
        f_card, _ = fit_scorecard(X_tr.iloc[fold_tr], y_tr.iloc[fold_tr])
        cv_scores.append(
            ev.metrics(y_tr.iloc[fold_va], f_card.predict_proba_bad(X_tr.iloc[fold_va]))["auc"]
        )
    cv_scores = np.array(cv_scores)

    print()
    print(ev.format_metrics("train", m_tr))
    print(ev.format_metrics("test", m_te))
    print(f"cv       AUC {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}  "
          f"over {args.cv_folds} folds")

    if not np.allclose(card.score(X_te), card.score_from_points(X_te)):
        raise RuntimeError("points table does not reconcile with the model")
    print("\npoints table reconciles with the model")

    scores_te = card.score(X_te)
    print(f"score range {scores_te.min():.0f} to {scores_te.max():.0f}, "
          f"median {scores_te.median():.0f}")

    tiers = scores_te.map(tier_for)
    tier_summary = pd.DataFrame({"tier": tiers, "bad": y_te}).groupby("tier").agg(
        n=("bad", "size"), bad=("bad", "sum")
    )
    tier_summary["share"] = tier_summary["n"] / len(y_te)
    tier_summary["bad_rate"] = tier_summary["bad"] / tier_summary["n"]
    print("\ntier cutoffs " + ", ".join(f"{n} >= {p:.0f}" for p, n in tier_cutoffs()))
    print(tier_summary.round(4).to_string())
    print("\ndecile table on the test set")
    print(ev.ks_table(y_te, scores_te).round(4).to_string())
    print("\ncalibration on the test set")
    print(ev.calibration_table(y_te, p_te).round(5).to_string())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "scorecard": card,
            "features": cfg.MODEL_FEATURES,
            "metrics": {"train": m_tr, "test": m_te,
                        "cv_auc_mean": float(cv_scores.mean()),
                        "cv_auc_std": float(cv_scores.std())},
            "scaling": {"base_score": cfg.BASE_SCORE, "base_odds": cfg.BASE_ODDS,
                        "pdo": cfg.PDO},
        },
        args.out,
    )
    print(f"\nwrote {args.out.relative_to(ROOT)} "
          f"({args.out.stat().st_size / 1024:.0f} KB)")

    report = args.out.with_suffix(".metrics.json")
    report.write_text(json.dumps({"train": m_tr, "test": m_te,
                                  "cv_auc_mean": float(cv_scores.mean()),
                                  "cv_auc_std": float(cv_scores.std())}, indent=2))
    print(f"wrote {report.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
