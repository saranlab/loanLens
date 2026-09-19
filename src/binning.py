"""Binning and WoE encoding, fit on training rows only.

This is where leakage would happen if the ordering were wrong. Bin membership
comes from fixed edges in config, so it carries no information, but the WoE value
attached to each bin is computed from the target. Fitting that on rows which
later serve as the test set inflates every metric reported afterwards.

Note there is no imputation step anywhere in the pipeline. WoE binning makes it
unnecessary: a missing value lands in its own bin and receives the WoE of the
group that is actually missing. Section 3 of the notebook found missingness has a
*lower* bad rate than average (5.61% against 6.95% for income), so replacing it
with the median would blend that group into the population and discard a real
signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import config as cfg

MISSING = "__missing__"


def bin_labels(values: pd.Series, feature: str) -> pd.Series:
    """Map a column to string bin labels, deterministically.

    Continuous features use the fixed edges from config. Discrete counters are
    capped and each remaining integer is its own bin. Flags pass through. Missing
    values always land in their own bin rather than being dropped.

    Kept as a module-level function so fit and transform cannot drift apart.
    """
    if feature in cfg.BIN_EDGES:
        cut = pd.cut(values, cfg.BIN_EDGES[feature])
        labels = cut.astype("object")
    elif feature in cfg.CLIP_AT:
        labels = values.clip(upper=cfg.CLIP_AT[feature]).astype("object")
    else:
        labels = values.astype("object")

    labels = labels.where(pd.notna(labels), MISSING)
    return labels.astype(str)


def expected_labels(feature: str) -> list[str] | None:
    """Every label a fixed-edge feature can produce, in order.

    Returns None for features whose bins come from the data rather than from
    config, since those cannot be enumerated ahead of time.
    """
    if feature not in cfg.BIN_EDGES:
        return None
    intervals = pd.IntervalIndex.from_breaks(cfg.BIN_EDGES[feature], closed="right")
    return [str(i) for i in intervals]


class WoEBinner(BaseEstimator, TransformerMixin):
    """Replace each feature with the WoE of the bin its value falls into.

    WoE is ln(%good / %bad), so a positive value means safer than the portfolio
    average and the sign runs the same direction as the final credit score.

    Parameters
    ----------
    features : list of column names to encode.
    eps : Haldane correction added to both counts, so a bin containing no bads
        yields a finite WoE rather than negative infinity.
    min_iv : features whose IV falls below this are dropped at fit time.
    min_bin_count : bins with fewer training rows than this are scored at the
        portfolio average (WoE 0) instead of getting their own estimate, and are
        listed in `thin_bins_`. Saying "no different from average" is the honest
        answer when a handful of rows is all there is.

    Attributes set by fit
    ---------------------
    woe_ : {feature: {bin_label: woe}}
    iv_ : {feature: information value}
    features_in_ : the features offered to fit
    features_ : the features that survived the IV filter
    tables_ : per-bin counts, kept for the scorecard and for audit
    thin_bins_ : {feature: [bin labels forced to WoE 0 for lack of rows]}
    """

    def __init__(
        self,
        features: list[str] | None = None,
        *,
        eps: float = cfg.WOE_EPS,
        min_iv: float = cfg.MIN_IV,
        min_bin_count: int = cfg.MIN_BIN_COUNT,
    ):
        self.features = features
        self.eps = eps
        self.min_iv = min_iv
        self.min_bin_count = min_bin_count

    def fit(self, X: pd.DataFrame, y) -> "WoEBinner":
        features = list(self.features) if self.features is not None else list(X.columns)
        missing = [f for f in features if f not in X.columns]
        if missing:
            raise ValueError(f"features not present in X: {missing}")

        y = pd.Series(np.asarray(y), index=X.index, name="target")
        if not set(y.unique()) <= {0, 1}:
            raise ValueError("y must be binary 0/1")

        self.features_in_ = features
        self.woe_, self.iv_, self.tables_ = {}, {}, {}
        self.thin_bins_ = {}

        total_bad = float(y.sum())
        total_good = float(len(y) - total_bad)
        if total_bad == 0 or total_good == 0:
            raise ValueError("y must contain both classes")

        for feature in features:
            labels = bin_labels(X[feature], feature)
            grouped = y.groupby(labels).agg(n="size", bad="sum")

            # groupby only yields bins that have rows. A configured bin with no
            # training rows still has to exist, or the exported scorecard has a
            # hole an applicant can fall into. Empty bins come back with n=0 and
            # are pinned to the portfolio average by the min_bin_count rule below,
            # which is the right answer: no data, no claim.
            expected = expected_labels(feature)
            if expected is not None:
                absent = [b for b in expected if b not in grouped.index]
                if absent:
                    grouped = pd.concat(
                        [grouped, pd.DataFrame({"n": 0, "bad": 0}, index=absent)]
                    )
                grouped = grouped.loc[
                    expected + [i for i in grouped.index if i not in expected]
                ]

            grouped["good"] = grouped["n"] - grouped["bad"]

            n_bins = len(grouped)
            dist_bad = (grouped["bad"] + self.eps) / (total_bad + self.eps * n_bins)
            dist_good = (grouped["good"] + self.eps) / (total_good + self.eps * n_bins)

            grouped["bad_rate"] = grouped["bad"] / grouped["n"]
            grouped["share"] = grouped["n"] / len(y)
            grouped["woe"] = np.log(dist_good / dist_bad)

            # A bin with too few rows to estimate from is pinned to the portfolio
            # average, so it neither helps nor penalises. Its IV contribution goes
            # to zero with it, which is correct: it carries no usable information.
            thin = grouped["n"] < self.min_bin_count
            if thin.any():
                grouped.loc[thin, "woe"] = 0.0
                self.thin_bins_[feature] = grouped.index[thin].tolist()

            grouped["iv_part"] = (dist_good - dist_bad) * grouped["woe"]

            self.woe_[feature] = grouped["woe"].to_dict()
            self.iv_[feature] = float(grouped["iv_part"].sum())
            self.tables_[feature] = grouped

        self.features_ = [f for f in features if self.iv_[f] >= self.min_iv]
        self.dropped_ = {f: self.iv_[f] for f in features if f not in self.features_}
        if not self.features_:
            raise ValueError(f"every feature fell below min_iv={self.min_iv}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "features_")
        out = {}
        self.unseen_bins_ = {}
        for feature in self.features_:
            labels = bin_labels(X[feature], feature)
            mapping = self.woe_[feature]
            encoded = labels.map(mapping)

            # A bin present at scoring time but absent from training gets WoE 0,
            # which is the portfolio average, rather than propagating NaN into the
            # model. Recorded so a monitoring job can alert on it.
            unseen = encoded.isna()
            if unseen.any():
                self.unseen_bins_[feature] = sorted(labels[unseen].unique().tolist())
                encoded = encoded.fillna(0.0)

            out[f"woe__{feature}"] = encoded.astype("float64")

        return pd.DataFrame(out, index=X.index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "features_")
        return np.asarray([f"woe__{f}" for f in self.features_], dtype=object)

    def iv_summary(self) -> pd.DataFrame:
        """IV per feature with the conventional strength reading, worst last."""
        check_is_fitted(self, "features_")
        out = pd.DataFrame(
            {"iv": pd.Series(self.iv_), "kept": pd.Series(self.iv_).index.isin(self.features_)}
        ).sort_values("iv", ascending=False)
        out["strength"] = pd.cut(
            out["iv"],
            [-np.inf, 0.02, 0.1, 0.3, 0.5, np.inf],
            labels=["useless", "weak", "medium", "strong", "very strong"],
        )
        out["monotonic"] = [self.is_monotonic(f) for f in out.index]
        return out

    def is_monotonic(self, feature: str) -> bool | None:
        """Does WoE move in one direction across the bins?

        None when the answer is not meaningful: unordered bins, or a feature whose
        only other bin is the missing one. A monotonic WoE curve is what a
        scorecard wants; section 7 found two features that are U shaped instead,
        which is the reason they are binned rather than fed in raw.
        """
        check_is_fitted(self, "features_")
        table = self.tables_[feature].drop(index=MISSING, errors="ignore")
        # A pinned bin carries no estimate, so it should not decide whether the
        # curve runs in one direction.
        pinned = self.thin_bins_.get(feature, [])
        table = table.drop(index=[b for b in pinned if b in table.index], errors="ignore")
        if len(table) < 3:
            return None
        woe = table["woe"].to_numpy()
        diffs = np.diff(woe)
        return bool(np.all(diffs >= 0) or np.all(diffs <= 0))

    def bin_table(self, feature: str) -> pd.DataFrame:
        """Per-bin counts, bad rate and WoE for one feature."""
        check_is_fitted(self, "features_")
        cols = ["n", "share", "bad", "bad_rate", "woe", "iv_part"]
        return self.tables_[feature][cols]
