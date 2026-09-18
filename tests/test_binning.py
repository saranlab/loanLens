"""Tests for WoE encoding, with the leakage boundary as the main target.

The synthetic frames here are deliberately small, so they pass min_bin_count=0 to
switch off the thin-bin floor. TestThinBins covers that floor on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src.binning import MISSING, WoEBinner, bin_labels


def make_frame(bad_by_bin: dict[float, int], per_bin: int = 50, col: str = "flag"):
    """Build a frame where each distinct value of `col` has a chosen bad count."""
    values, targets = [], []
    for value, n_bad in bad_by_bin.items():
        values += [value] * per_bin
        targets += [1] * n_bad + [0] * (per_bin - n_bad)
    return pd.DataFrame({col: values}), pd.Series(targets, name=cfg.TARGET)


class TestWoEFormula:
    def test_matches_a_hand_computed_value(self):
        """Pins the formula, the Haldane correction and the sign convention.

        Two bins of 50. Bin 0 has 5 bad, bin 1 has 15, so 20 bad and 80 good
        overall. With eps=0.5 and 2 bins:
            dist_good(0) = (45 + 0.5) / (80 + 0.5 * 2) = 45.5 / 81
            dist_bad(0)  = (5 + 0.5)  / (20 + 0.5 * 2) = 5.5 / 21
            ratio        = (45.5 * 21) / (81 * 5.5) = 955.5 / 445.5 = 2.1447811
            woe(0)       = ln(2.1447811) = 0.7630375
        """
        X, y = make_frame({0.0: 5, 1.0: 15})
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X, y)
        assert binner.woe_["flag"]["0.0"] == pytest.approx(0.7630375, abs=1e-6)

    def test_safer_bin_gets_positive_woe(self):
        """Sign convention: ln(%good / %bad), so above-average safety is positive."""
        X, y = make_frame({0.0: 5, 1.0: 15})
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X, y)
        assert binner.woe_["flag"]["0.0"] > 0
        assert binner.woe_["flag"]["1.0"] < 0

    def test_empty_bin_stays_finite(self):
        """Without the Haldane correction this bin would give -inf."""
        X, y = make_frame({0.0: 0, 1.0: 25})
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X, y)
        assert np.isfinite(binner.woe_["flag"]["0.0"])

    def test_iv_is_unaffected_by_sign_convention(self):
        """IV flips both factors of the product, so it comes out the same."""
        X, y = make_frame({0.0: 5, 1.0: 15})
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X, y)
        iv_good_over_bad = binner.iv_["flag"]

        table = binner.tables_["flag"]
        n_bins = len(table)
        total_bad, total_good = float(y.sum()), float(len(y) - y.sum())
        dist_bad = (table["bad"] + 0.5) / (total_bad + 0.5 * n_bins)
        dist_good = (table["good"] + 0.5) / (total_good + 0.5 * n_bins)
        flipped = float(((dist_bad - dist_good) * np.log(dist_bad / dist_good)).sum())

        assert iv_good_over_bad == pytest.approx(flipped)


class TestLeakageBoundary:
    def test_woe_comes_from_fit_rows_only(self):
        """The same test rows encode differently depending on what fit saw.

        This is the whole reason binning has to come after the split: if fit sees
        the test rows, their own outcomes shape the values they get encoded with.
        """
        X_train, y_train = make_frame({0.0: 5, 1.0: 15})
        X_test, y_test = make_frame({0.0: 40, 1.0: 2}, per_bin=50)

        train_only = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X_train, y_train)
        pooled = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(
            pd.concat([X_train, X_test], ignore_index=True),
            pd.concat([y_train, y_test], ignore_index=True),
        )

        assert not np.allclose(
            train_only.transform(X_test).to_numpy(),
            pooled.transform(X_test).to_numpy(),
        )

    def test_transform_needs_no_target(self):
        X_train, y_train = make_frame({0.0: 5, 1.0: 15})
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X_train, y_train)
        # scoring a single applicant, with no outcome to look up
        one = pd.DataFrame({"flag": [1.0]})
        assert binner.transform(one).shape == (1, 1)

    def test_row_order_does_not_change_encoding(self):
        X, y = make_frame({0.0: 5, 1.0: 15})
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X, y)
        shuffled = X.sample(frac=1.0, random_state=0)
        pd.testing.assert_frame_equal(
            binner.transform(shuffled).sort_index(),
            binner.transform(X).sort_index(),
        )


class TestMissingValues:
    def test_missing_is_its_own_bin_not_imputed(self):
        X = pd.DataFrame({"flag": [0.0] * 40 + [1.0] * 40 + [np.nan] * 20})
        y = pd.Series([1] * 4 + [0] * 36 + [1] * 12 + [0] * 28 + [1] * 1 + [0] * 19)
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X, y)

        assert MISSING in binner.woe_["flag"]
        assert binner.tables_["flag"].loc[MISSING, "n"] == 20
        # 5% bad against 17% overall, so the missing group reads as safer
        assert binner.woe_["flag"][MISSING] > 0

    def test_missing_rows_are_encoded_not_dropped(self):
        X = pd.DataFrame({"flag": [0.0] * 40 + [1.0] * 40 + [np.nan] * 20})
        y = pd.Series([0, 1] * 50)
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X, y)
        out = binner.transform(X)
        assert len(out) == len(X)
        assert out.notna().all().all()


class TestUnseenBins:
    def test_unseen_bin_scores_neutral_and_is_recorded(self):
        X_train, y_train = make_frame({0.0: 5, 1.0: 15})
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X_train, y_train)

        surprise = pd.DataFrame({"flag": [0.0, 1.0, 7.0]})
        out = binner.transform(surprise)

        assert out.iloc[2, 0] == 0.0
        assert binner.unseen_bins_["flag"] == ["7.0"]

    def test_no_unseen_bins_recorded_when_all_known(self):
        X_train, y_train = make_frame({0.0: 5, 1.0: 15})
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X_train, y_train)
        binner.transform(X_train)
        assert binner.unseen_bins_ == {}


class TestFeatureSelection:
    def test_low_iv_feature_is_dropped(self):
        rng = np.random.default_rng(0)
        n = 4000
        X = pd.DataFrame(
            {
                "signal": rng.integers(0, 2, n).astype(float),
                "noise": rng.integers(0, 2, n).astype(float),
            }
        )
        # signal drives the target, noise does not
        y = pd.Series((rng.random(n) < np.where(X["signal"] == 1, 0.4, 0.05)).astype(int))

        binner = WoEBinner(["signal", "noise"], min_iv=0.02, min_bin_count=0).fit(X, y)
        assert binner.features_ == ["signal"]
        assert "noise" in binner.dropped_
        assert list(binner.get_feature_names_out()) == ["woe__signal"]

    def test_raises_when_nothing_survives(self):
        X, y = make_frame({0.0: 10, 1.0: 10})
        with pytest.raises(ValueError, match="min_iv"):
            WoEBinner(["flag"], min_iv=0.5, min_bin_count=0).fit(X, y)

    def test_rejects_non_binary_target(self):
        X, _ = make_frame({0.0: 5, 1.0: 15})
        with pytest.raises(ValueError, match="binary"):
            WoEBinner(["flag"]).fit(X, pd.Series([0, 1, 2] * 33 + [0]))


class TestMonotonicity:
    def test_detects_monotonic_woe(self):
        X, y = make_frame({0.0: 5, 1.0: 15, 2.0: 30}, col="NumberOfDependents")
        binner = WoEBinner(["NumberOfDependents"], min_iv=0.0, min_bin_count=0).fit(X, y)
        assert binner.is_monotonic("NumberOfDependents") is True

    def test_detects_u_shape(self):
        """High at both ends, low in the middle: what section 7 found twice."""
        X, y = make_frame({0.0: 30, 1.0: 5, 2.0: 30}, col="NumberOfDependents")
        binner = WoEBinner(["NumberOfDependents"], min_iv=0.0, min_bin_count=0).fit(X, y)
        assert binner.is_monotonic("NumberOfDependents") is False

    def test_undecidable_with_too_few_bins(self):
        X, y = make_frame({0.0: 5, 1.0: 15})
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=0).fit(X, y)
        assert binner.is_monotonic("flag") is None


class TestBinLabels:
    def test_edges_are_right_closed(self):
        """A value exactly on an edge belongs to the lower bin."""
        s = pd.Series([0.1, 0.100001])
        labels = bin_labels(s, "RevolvingUtilizationOfUnsecuredLines")
        assert labels.iloc[0] != labels.iloc[1]
        assert "0.1]" in labels.iloc[0]

    def test_outer_edges_are_unbounded(self):
        """Nothing can fall outside the bins, so scoring never sees a gap."""
        s = pd.Series([-999.0, 1e9])
        labels = bin_labels(s, "MonthlyIncome")
        assert MISSING not in labels.tolist()

    def test_counters_are_capped(self):
        s = pd.Series([0, 3, 4, 99])
        labels = bin_labels(s, "NumberOfTimes90DaysLate")
        assert labels.tolist() == ["0", "3", "3", "3"]

    def test_labels_are_row_independent(self):
        s = pd.Series([0.05, 0.4, np.nan, 1.5, 20.0])
        full = bin_labels(s, "RevolvingUtilizationOfUnsecuredLines")
        subset = bin_labels(s.iloc[[1, 3]], "RevolvingUtilizationOfUnsecuredLines")
        assert subset.tolist() == full.iloc[[1, 3]].tolist()


class TestThinBins:
    def test_bin_below_floor_is_pinned_to_average(self):
        """A 5-row bin gets WoE 0 rather than a rate estimated from 5 rows."""
        X = pd.DataFrame({"flag": [0.0] * 500 + [1.0] * 500 + [2.0] * 5})
        y = pd.Series([1] * 25 + [0] * 475 + [1] * 150 + [0] * 350 + [1] * 5)

        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=100).fit(X, y)

        assert binner.woe_["flag"]["2.0"] == 0.0
        assert binner.thin_bins_["flag"] == ["2.0"]
        # the bins that do have enough rows still get real estimates
        assert binner.woe_["flag"]["0.0"] > 0
        assert binner.woe_["flag"]["1.0"] < 0

    def test_pinned_bin_adds_nothing_to_iv(self):
        X = pd.DataFrame({"flag": [0.0] * 500 + [1.0] * 500 + [2.0] * 5})
        y = pd.Series([1] * 25 + [0] * 475 + [1] * 150 + [0] * 350 + [1] * 5)
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=100).fit(X, y)
        assert binner.tables_["flag"].loc["2.0", "iv_part"] == 0.0

    def test_nothing_pinned_when_every_bin_is_large_enough(self):
        X = pd.DataFrame({"flag": [0.0] * 500 + [1.0] * 500})
        y = pd.Series([1] * 25 + [0] * 475 + [1] * 150 + [0] * 350)
        binner = WoEBinner(["flag"], min_iv=0.0, min_bin_count=100).fit(X, y)
        assert binner.thin_bins_ == {}
