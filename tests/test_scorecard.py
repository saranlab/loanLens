"""Tests for the PDO scale and the points conversion.

The direction tests matter most. A sign error here produces a scorecard that
looks entirely reasonable and ranks applicants backwards, and nothing in the
pipeline raises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config as cfg
from src.binning import WoEBinner
from src.scorecard import (
    Scorecard,
    odds_to_score,
    probability_to_score,
    scaling_constants,
    score_to_odds,
    tier_cutoffs,
    tier_for,
)


class TestScaling:
    def test_factor_and_offset(self):
        """factor = PDO / ln(2); offset places base_score at base_odds."""
        factor, offset = scaling_constants(base_score=600, base_odds=50, pdo=20)
        assert factor == pytest.approx(28.853901, abs=1e-6)
        assert offset == pytest.approx(487.122876, abs=1e-6)

    def test_base_odds_give_base_score(self):
        assert odds_to_score(50.0) == pytest.approx(600.0)

    def test_pdo_doubles_the_odds(self):
        """20 points per doubling, which is what PDO means."""
        assert odds_to_score(50.0) == pytest.approx(600.0)
        assert odds_to_score(100.0) == pytest.approx(620.0)
        assert odds_to_score(200.0) == pytest.approx(640.0)
        assert odds_to_score(25.0) == pytest.approx(580.0)

    def test_round_trip(self):
        for score in [480.0, 600.0, 712.5, 800.0]:
            assert odds_to_score(score_to_odds(score)) == pytest.approx(score)

    def test_probability_route_agrees_with_odds_route(self):
        """p_bad = 1/51 is 50:1 good:bad, so it must land on the base score."""
        assert probability_to_score(1.0 / 51.0) == pytest.approx(600.0)

    def test_extreme_probabilities_stay_finite(self):
        assert np.isfinite(probability_to_score(0.0))
        assert np.isfinite(probability_to_score(1.0))

    def test_lower_default_probability_scores_higher(self):
        assert probability_to_score(0.01) > probability_to_score(0.50)

    def test_custom_pdo_changes_the_spacing(self):
        assert odds_to_score(100.0, pdo=40) == pytest.approx(640.0)


class TestTiers:
    def test_cutoffs_come_from_the_pd_boundaries(self):
        """Tier A caps PD at 1%, tier B at 5%. The points follow from the scale."""
        cuts = dict((name, pts) for pts, name in tier_cutoffs())
        assert cuts["A"] == pytest.approx(probability_to_score(0.01))
        assert cuts["B"] == pytest.approx(probability_to_score(0.05))

    def test_boundaries_are_inclusive_at_the_bottom_of_each_tier(self):
        a_cut, b_cut = (pts for pts, _ in tier_cutoffs())
        assert tier_for(a_cut) == "A"
        assert tier_for(a_cut - 0.01) == "B"
        assert tier_for(b_cut) == "B"
        assert tier_for(b_cut - 0.01) == "C"

    def test_far_ends(self):
        assert tier_for(900) == "A"
        assert tier_for(0) == "C"

    def test_tier_a_is_reachable_on_this_scale(self):
        """The 720 cutoff the original plan proposed needed a PD of 0.03%.

        Nobody in this portfolio is that safe, so tier A would have been empty.
        Deriving the cutoff keeps it inside the range the model can produce.
        """
        a_cut = tier_cutoffs()[0][0]
        assert a_cut < 660

    def test_cutoffs_move_with_the_scale(self):
        """Change the scaling and the cutoffs follow instead of stranding."""
        shifted = tier_cutoffs(base_score=300, base_odds=20, pdo=40)
        assert shifted[0][0] != tier_cutoffs()[0][0]


@pytest.fixture
def fitted():
    """A two-feature scorecard on synthetic data with a known risk ordering.

    `flag` 0 is the safe group (2% bad), 1 is the risky group (30% bad).
    `counter` rises in risk with its value.
    """
    rng = np.random.default_rng(7)
    n = 6000
    flag = rng.integers(0, 2, n).astype(float)
    counter = rng.integers(0, 3, n).astype(float)
    p_bad = np.where(flag == 1, 0.30, 0.02) + 0.05 * counter
    y = pd.Series((rng.random(n) < p_bad).astype(int))
    X = pd.DataFrame({"flag": flag, "NumberOfDependents": counter})

    binner = WoEBinner(["flag", "NumberOfDependents"], min_iv=0.0).fit(X, y)
    model = LogisticRegression().fit(binner.transform(X), y)
    return Scorecard(binner, model), X, y


class TestDirection:
    def test_woe_coefficients_come_out_negative(self):
        """WoE is ln(good/bad), so a higher WoE must lower P(bad).

        If this flips, the WoE convention and the score conversion have drifted
        apart. Asserted separately from the score direction so a failure says
        which of the two moved.
        """
        rng = np.random.default_rng(1)
        n = 4000
        flag = rng.integers(0, 2, n).astype(float)
        y = pd.Series((rng.random(n) < np.where(flag == 1, 0.3, 0.02)).astype(int))
        X = pd.DataFrame({"flag": flag})
        binner = WoEBinner(["flag"], min_iv=0.0).fit(X, y)
        model = LogisticRegression().fit(binner.transform(X), y)
        assert model.coef_.ravel()[0] < 0

    def test_safer_applicant_scores_higher(self, fitted):
        card, _, _ = fitted
        safe = pd.DataFrame({"flag": [0.0], "NumberOfDependents": [0.0]})
        risky = pd.DataFrame({"flag": [1.0], "NumberOfDependents": [2.0]})
        assert card.score(safe).iloc[0] > card.score(risky).iloc[0]

    def test_safer_bin_carries_more_points(self, fitted):
        card, _, _ = fitted
        table = card.points_table().set_index(["feature", "bin"])
        assert table.loc[("flag", "0.0"), "points"] > table.loc[("flag", "1.0"), "points"]

    def test_points_rank_the_same_way_as_bad_rate(self, fitted):
        """Across every bin of every feature, more points means lower bad rate."""
        table = fitted[0].points_table()
        for feature, group in table.groupby("feature"):
            if len(group) < 2:
                continue
            corr = group["points"].corr(group["bad_rate"])
            assert corr < 0, feature


class TestPointsReconcile:
    def test_two_score_routes_agree(self, fitted):
        """Summing the points table must equal scoring through the model.

        If these drift, the table a human reads is not the model that decided.
        """
        card, X, _ = fitted
        pd.testing.assert_series_equal(
            card.score(X), card.score_from_points(X), rtol=1e-9
        )

    def test_points_table_covers_every_bin(self, fitted):
        card, _, _ = fitted
        table = card.points_table()
        for feature in card.features:
            assert len(table[table["feature"] == feature]) == len(
                card.binner.tables_[feature]
            )

    def test_base_points_is_the_score_at_zero_woe(self, fitted):
        card, _, _ = fitted
        zeroed = pd.DataFrame(
            {f"woe__{f}": [0.0] for f in card.features}
        )
        p = card.model.predict_proba(zeroed)[:, 1]
        assert probability_to_score(p)[0] == pytest.approx(card.base_points)

    def test_mismatched_coefficient_count_is_rejected(self, fitted):
        card, X, y = fitted
        stunted = LogisticRegression().fit(
            card.binner.transform(X).iloc[:, :1], y
        )
        with pytest.raises(ValueError, match="coefficients"):
            Scorecard(card.binner, stunted)


class TestExplain:
    def test_names_the_costliest_feature_first(self, fitted):
        card, _, _ = fitted
        worst = pd.DataFrame({"flag": [1.0], "NumberOfDependents": [2.0]})
        reasons = card.explain(worst)["reasons"].iloc[0]
        assert reasons[0].startswith("flag")

    def test_best_possible_applicant_has_no_real_shortfall(self, fitted):
        card, _, _ = fitted
        best = pd.DataFrame({"flag": [0.0], "NumberOfDependents": [0.0]})
        reasons = card.explain(best)["reasons"].iloc[0]
        deductions = [float(r.split("(-")[1].rstrip(")")) for r in reasons]
        assert max(deductions) == pytest.approx(0.0, abs=1e-9)

    def test_returns_a_row_per_applicant(self, fitted):
        card, X, _ = fitted
        out = card.explain(X.head(20))
        assert len(out) == 20
        assert list(out.columns) == ["score", "reasons"]
