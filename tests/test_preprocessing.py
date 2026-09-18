"""Tests for the cleaning rules.

Uses a small hand-built frame rather than the Kaggle CSVs, so the suite runs on
a fresh clone where data/ is empty.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.preprocessing import add_features, clean

UTIL = "RevolvingUtilizationOfUnsecuredLines"
LATE_30, LATE_90, LATE_60 = cfg.LATE_COLS


@pytest.fixture
def frame() -> pd.DataFrame:
    """Eight rows, each one an edge case from notebook section 5."""
    return pd.DataFrame(
        {
            cfg.TARGET: [0, 1, 0, 1, 0, 0, 1, 0],
            UTIL: [0.25, 1.40, 50708.0, 0.90, 13.0, 14.0, 0.5, 0.0],
            "age": [45, 21, 0, 70, 52, 33, 17, 18],
            LATE_30: [0, 2, 98, 1, 96, 0, 3, 0],
            "DebtRatio": [0.35, 0.80, 2449.0, 0.12, 1.50, 5000.0, 0.44, 0.0],
            "MonthlyIncome": [5400.0, 2600.0, np.nan, 9120.0, 3000.0, np.nan, 0.0, 1200.0],
            "NumberOfOpenCreditLinesAndLoans": [8, 4, 13, 2, 6, 9, 11, 1],
            LATE_90: [0, 0, 98, 0, 96, 0, 1, 0],
            "NumberRealEstateLoansOrLines": [1, 0, 6, 2, 1, 3, 0, 0],
            LATE_60: [0, 1, 98, 0, 96, 0, 0, 0],
            "NumberOfDependents": [2.0, 1.0, np.nan, 0.0, 3.0, np.nan, 1.0, 0.0],
        }
    )


def test_no_rows_dropped(frame):
    assert len(clean(frame)) == len(frame)


def test_input_not_mutated(frame):
    before = frame.copy(deep=True)
    clean(frame)
    pd.testing.assert_frame_equal(frame, before)


class TestSentinelCodes:
    def test_codes_become_nan_in_all_three_counters(self, frame):
        out = clean(frame)
        for col in cfg.LATE_COLS:
            assert out[col].isna().sum() == 2, col
        assert not out[cfg.LATE_COLS].isin(cfg.SENTINEL_CODES).any().any()

    def test_flag_marks_exactly_the_coded_rows(self, frame):
        out = clean(frame)
        assert out["late_code_flag"].tolist() == [0, 0, 1, 0, 1, 0, 0, 0]

    def test_genuine_counts_survive(self, frame):
        """A borrower with 3 real late payments must not be confused with a code."""
        out = clean(frame)
        assert out[LATE_30].iloc[6] == 3
        assert out[LATE_30].iloc[1] == 2


class TestUtilization:
    def test_over_limit_is_kept(self, frame):
        """1.40 is real risk (40.1% bad rate in section 5.2), not an error."""
        out = clean(frame)
        assert out[UTIL].iloc[1] == pytest.approx(1.40)

    def test_absurd_values_become_nan(self, frame):
        out = clean(frame)
        assert np.isnan(out[UTIL].iloc[2])
        assert np.isnan(out[UTIL].iloc[5])

    def test_threshold_is_exclusive(self, frame):
        """Exactly at the threshold stays; strictly above it goes."""
        out = clean(frame)
        assert out[UTIL].iloc[4] == pytest.approx(cfg.UTIL_INVALID_ABOVE)
        assert out["util_invalid_flag"].tolist() == [0, 0, 1, 0, 0, 1, 0, 0]


class TestDebtRatioSplit:
    def test_split_is_mutually_exclusive(self, frame):
        out = clean(frame)
        assert not (out["debt_ratio"].notna() & out["debt_amount"].notna()).any()

    def test_split_is_lossless(self, frame):
        """Every original value lands in exactly one of the two columns."""
        out = clean(frame)
        recombined = out["debt_ratio"].fillna(out["debt_amount"])
        pd.testing.assert_series_equal(
            recombined, frame["DebtRatio"], check_names=False
        )

    def test_amounts_go_with_missing_income(self, frame):
        out = clean(frame)
        assert out.loc[out["income_missing"] == 1, "debt_amount"].notna().all()
        assert out.loc[out["income_missing"] == 0, "debt_amount"].isna().all()


class TestImpossibleValues:
    def test_under_age_becomes_nan(self, frame):
        out = clean(frame)
        assert np.isnan(out["age"].iloc[2])
        assert np.isnan(out["age"].iloc[6])

    def test_minimum_age_is_inclusive(self, frame):
        out = clean(frame)
        assert out["age"].iloc[7] == cfg.MIN_AGE

    def test_zero_income_is_flagged_not_nulled(self, frame):
        """Zero income can be true, so it is marked rather than removed."""
        out = clean(frame)
        assert out["MonthlyIncome"].iloc[6] == 0.0
        assert out["income_zero"].iloc[6] == 1


def test_clean_is_row_independent(frame):
    """Cleaning a subset equals cleaning everything and then subsetting.

    This is what makes the transformer leak-proof: no row's output depends on any
    other row, so putting it before the train/test split cannot move information
    across the boundary.
    """
    rows = [1, 2, 5]
    from_subset = clean(frame.iloc[rows]).reset_index(drop=True)
    from_whole = clean(frame).iloc[rows].reset_index(drop=True)
    pd.testing.assert_frame_equal(from_subset, from_whole)


def test_add_features_is_row_independent(frame):
    rows = [0, 3, 6]
    base = clean(frame)
    from_subset = add_features(base.iloc[rows]).reset_index(drop=True)
    from_whole = add_features(base).iloc[rows].reset_index(drop=True)
    pd.testing.assert_frame_equal(from_subset, from_whole)


class TestDerivedFeatures:
    def test_weighted_late_applies_severity(self, frame):
        out = add_features(clean(frame))
        # row 1: two 30-day and one 60-day event -> 1*2 + 2*1 + 3*0
        assert out["weighted_late"].iloc[1] == 4
        # row 6: three 30-day and one 90-day -> 1*3 + 2*0 + 3*1
        assert out["weighted_late"].iloc[6] == 6

    def test_sentinel_rows_give_nan_not_a_huge_count(self, frame):
        """The whole point of nulling 98 first: 98+98+98 must not become 294."""
        out = add_features(clean(frame))
        assert np.isnan(out["weighted_late"].iloc[2])
        assert np.isnan(out["total_late"].iloc[2])

    def test_rejected_candidates_are_not_rebuilt(self, frame):
        out = add_features(clean(frame))
        for col in ["util_per_line", "non_realestate_lines", "family_size_group"]:
            assert col not in out.columns

    def test_real_estate_share_handles_zero_accounts(self, frame):
        out = add_features(clean(frame))
        assert out["real_estate_share"].notna().all()
