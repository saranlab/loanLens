"""Gradient boosting baseline, to price what staying interpretable costs.

    python -m src.baseline

The scorecard is deliberately constrained: monotonic WoE bins, a linear model,
one row per feature that a declined applicant could have explained to them. Those
constraints cost accuracy. This module measures how much, so the choice can be
stated as a trade rather than assumed to be free.

The comparison is set up to be fair to the tree, not to the scorecard:

- It gets the same train/test split, from the same function.
- It gets the features the scorecard dropped for interpretability, not for lack of
  signal: `weighted_late` (IV 1.51 against 0.88 for the best single counter) and
  `disposable_income` (0.126 against 0.074).
- It gets raw values, since trees do not need WoE and binning would only throw
  resolution away.
- Early stopping runs against a slice carved out of train, never the test set.

If the tree still wins by a wide margin, that is the honest cost of the scorecard.
If it barely wins, the constraints were close to free and the scorecard is simply
the better choice.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as cfg
from . import evaluate as ev
from .train import DEFAULT_TRAIN_CSV, ROOT, build_dataset, fit_scorecard, split

# Columns the tree is not allowed to see. debt_amount and income_zero are
# derived from the same DebtRatio split the scorecard uses, so they stay, but
# raw DebtRatio is excluded: it mixes two units and the split columns replace it.
EXCLUDE = ["DebtRatio"]


def tree_features(X: pd.DataFrame) -> list[str]:
    """Every numeric column except the ones that duplicate a split column."""
    numeric = X.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in EXCLUDE]


def fit_tree(X_tr: pd.DataFrame, y_tr: pd.Series, features: list[str], seed: int):
    """LightGBM with early stopping against a validation slice cut from train."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr[features], y_tr, test_size=0.2, stratify=y_tr, random_state=seed
    )
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=100,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        verbose=-1,
    )
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_TRAIN_CSV)
    parser.add_argument("--seeds", type=int, default=3,
                        help="repeat both models over this many seeds")
    args = parser.parse_args(argv)

    if not args.csv.exists():
        parser.error(f"{args.csv} not found. See README for the Kaggle download.")

    X, y = build_dataset(args.csv)
    X_tr, X_te, y_tr, y_te = split(X, y)
    features = tree_features(X_tr)

    print(f"train {len(X_tr):,}   test {len(X_te):,}")
    print(f"scorecard features: {len(cfg.MODEL_FEATURES)}")
    print(f"tree features:      {len(features)}")
    extra = [f for f in features if f not in cfg.MODEL_FEATURES]
    print(f"  the tree also sees: {', '.join(extra)}")

    rows = []
    for i in range(args.seeds):
        seed = cfg.RANDOM_STATE + i

        card, _ = fit_scorecard(X_tr, y_tr)
        card_metrics = ev.metrics(y_te, card.predict_proba_bad(X_te))

        tree = fit_tree(X_tr, y_tr, features, seed)
        tree_metrics = ev.metrics(y_te, tree.predict_proba(X_te[features])[:, 1])

        rows.append({"seed": seed, "model": "scorecard", **card_metrics})
        rows.append({"seed": seed, "model": "lightgbm", "trees": tree.best_iteration_,
                     **tree_metrics})

    results = pd.DataFrame(rows)
    summary = results.groupby("model")[["auc", "gini", "ks", "pr_auc", "brier"]].agg(
        ["mean", "std"]
    )
    print("\nheld-out test set, averaged over seeds")
    print(summary.round(4).to_string())

    card_auc = results.loc[results.model == "scorecard", "auc"].mean()
    tree_auc = results.loc[results.model == "lightgbm", "auc"].mean()
    card_ks = results.loc[results.model == "scorecard", "ks"].mean()
    tree_ks = results.loc[results.model == "lightgbm", "ks"].mean()

    print(
        f"\ninterpretability costs {tree_auc - card_auc:.4f} AUC "
        f"({(tree_auc - card_auc) / card_auc:.2%} relative) "
        f"and {tree_ks - card_ks:.4f} KS"
    )

    # The scorecard is deterministic given the split, so its spread across seeds
    # is zero by construction. Only the tree's spread is informative.
    tree_std = results.loc[results.model == "lightgbm", "auc"].std()
    if args.seeds > 1 and tree_std > 0:
        print(f"the tree's own seed-to-seed spread is {tree_std:.4f} AUC, "
              f"so read the gap against that")

    print("\ntop tree features by gain")
    tree = fit_tree(X_tr, y_tr, features, cfg.RANDOM_STATE)
    gains = pd.Series(tree.booster_.feature_importance("gain"), index=features)
    print((gains / gains.sum()).sort_values(ascending=False).head(10).round(4).to_string())

    out = ROOT / "artifacts" / "baseline_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
