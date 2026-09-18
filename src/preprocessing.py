"""Cleaning and feature construction, lifted out of the EDA notebook.

Two rules hold throughout:

1. No rows are dropped. Every value that gets nulled leaves a flag behind, because
   the EDA showed the anomalies themselves carry risk information (the 96/98
   status codes have a 54.7% bad rate).
2. Nothing here looks at the target or at any statistic pooled across rows, so
   these transformers are stateless and cannot leak. Anything that has to be
   estimated from data (bin edges, WoE, imputation values) lives in binning.py
   and is fit on the training fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config as cfg


def clean(data: pd.DataFrame) -> pd.DataFrame:
    """Null sentinels and impossible values, keeping each signal as a flag.

    Row count is unchanged. See notebook sections 5.1 to 5.4 for the evidence
    behind each threshold.
    """
    out = data.copy()

    # 5.1 Bureau status codes in the delinquency counters.
    out["late_code_flag"] = (
        out[cfg.LATE_COLS].isin(cfg.SENTINEL_CODES).any(axis=1).astype("int8")
    )
    for col in cfg.LATE_COLS:
        out[col] = out[col].astype("float64")
        out.loc[out[col].isin(cfg.SENTINEL_CODES), col] = np.nan

    # 5.2 Utilization values whose bad rate contradicts the risk they imply.
    util = "RevolvingUtilizationOfUnsecuredLines"
    out["util_invalid_flag"] = (out[util] > cfg.UTIL_INVALID_ABOVE).astype("int8")
    out.loc[out[util] > cfg.UTIL_INVALID_ABOVE, util] = np.nan

    # 5.3 DebtRatio holds a ratio when income is known and an amount when it is
    # not, so it splits into two columns rather than being capped.
    out["income_missing"] = out["MonthlyIncome"].isna().astype("int8")
    out["debt_ratio"] = out["DebtRatio"].where(out["income_missing"] == 0)
    out["debt_amount"] = out["DebtRatio"].where(out["income_missing"] == 1)

    # 5.4 Impossible values.
    out["age"] = out["age"].astype("float64")
    out.loc[out["age"] < cfg.MIN_AGE, "age"] = np.nan
    out["income_zero"] = (out["MonthlyIncome"] == 0).astype("int8")

    return out


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Derived features that beat the IV of the column they came from.

    Section 10 built ten candidates and measured each against its baseline. The
    four that lost are not rebuilt here: util_per_line (0.875 vs 1.150, the
    numerator excludes installment debt while the denominator counts it),
    non_realestate_lines (0.064 vs 0.081), monthly_debt as a feature in its own
    right (0.036 vs 0.072, though it stays as an intermediate for
    disposable_income), and family_size_group (0.028 vs 0.036, and its top bin
    held 245 rows).

    Features the scorecard does not use are still built here, because the
    FastAPI service and any later tree model read from the same transform.
    """
    out = data.copy()

    # Delinquency history, summed and severity weighted. IV 1.51 against 0.88 for
    # the strongest single counter: the three correlate only 0.24 to 0.30, so each
    # holds something the others do not. The 1/2/3 weights are a judgement call,
    # not estimated from the data.
    out["total_late"] = out[cfg.LATE_COLS].sum(axis=1, min_count=1)
    out["weighted_late"] = (
        1 * out["NumberOfTime30-59DaysPastDueNotWorse"]
        + 2 * out["NumberOfTime60-89DaysPastDueNotWorse"]
        + 3 * out["NumberOfTimes90DaysLate"]
    )
    out["has_late"] = (out["total_late"] > 0).astype("int8")

    # Account mix, aimed at the U shape in section 7. IV 0.084 against 0.062 for
    # the raw count.
    out["real_estate_share"] = out["NumberRealEstateLoansOrLines"] / out[
        "NumberOfOpenCreditLinesAndLoans"
    ].replace(0, np.nan)

    # Repayment capacity. disposable_income reaches IV 0.126 against 0.074 for
    # MonthlyIncome alone, which follows from DebtRatio already covering living
    # costs.
    out["income_per_person"] = out["MonthlyIncome"] / (
        out["NumberOfDependents"].fillna(0) + 1
    )
    out["monthly_debt"] = out["debt_ratio"] * out["MonthlyIncome"]
    out["disposable_income"] = out["MonthlyIncome"] - out["monthly_debt"]

    return out


class CreditCleaner(BaseEstimator, TransformerMixin):
    """Stateless wrapper so clean() and add_features() can sit in a Pipeline."""

    def __init__(self, *, derive_features: bool = True):
        self.derive_features = derive_features

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = clean(X)
        if self.derive_features:
            out = add_features(out)
        return out

    def get_feature_names_out(self, input_features=None):
        raise NotImplementedError(
            "This transformer returns a DataFrame; read .columns off the result."
        )


def load_raw(path) -> pd.DataFrame:
    """Read a competition CSV, using its unnamed first column as the index."""
    df = pd.read_csv(path, index_col=0)
    df.index.name = "id"
    return df
