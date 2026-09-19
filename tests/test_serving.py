"""Tests for the JSON export and the dependency-light scorer.

The one that matters is TestParity. Splitting training from serving means binning
is implemented twice, and two implementations drift. These tests are what stop
that being discovered in production.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import config as cfg
from src.binning import WoEBinner
from src.scorecard import Scorecard
from src.serving import SCHEMA_VERSION, ScoringModel, export, frame_to_applicants, to_dict

# One feature of each kind the exporter has to handle.
FEATURES = ["age", "NumberOfTimes90DaysLate", "late_code_flag"]


@pytest.fixture
def fitted():
    """A scorecard over an interval feature, a clipped counter and a flag."""
    rng = np.random.default_rng(11)
    n = 8000
    X = pd.DataFrame(
        {
            "age": rng.integers(20, 85, n).astype(float),
            "NumberOfTimes90DaysLate": rng.integers(0, 6, n).astype(float),
            "late_code_flag": rng.integers(0, 2, n).astype("int8"),
        }
    )
    # missing values in both a binned and a counted feature
    X.loc[rng.random(n) < 0.05, "age"] = np.nan
    X.loc[rng.random(n) < 0.03, "NumberOfTimes90DaysLate"] = np.nan

    risk = (
        0.02
        + 0.06 * (X["NumberOfTimes90DaysLate"].fillna(0) > 0)
        + 0.05 * (X["age"].fillna(50) < 35)
        + 0.10 * X["late_code_flag"]
    )
    y = pd.Series((rng.random(n) < risk).astype(int), index=X.index)

    binner = WoEBinner(FEATURES, min_iv=0.0).fit(X, y)
    model = LogisticRegression(max_iter=1000).fit(binner.transform(X), y)
    return Scorecard(binner, model), X, y


class TestParity:
    def test_json_scorer_reproduces_the_model(self, fitted):
        """Every row, both paths, to floating point.

        This is the guard that makes it safe to serve JSON instead of the fitted
        object. If binning ever diverges between training and serving, this fails
        before anything reaches a container.
        """
        card, X, _ = fitted
        model = ScoringModel(to_dict(card))
        from_json = np.array([model.score(a) for a in frame_to_applicants(X, card.features)])
        np.testing.assert_allclose(from_json, card.score(X).to_numpy(), atol=1e-9)

    def test_probability_matches_too(self, fitted):
        card, X, _ = fitted
        model = ScoringModel(to_dict(card))
        from_json = np.array(
            [model.probability(a) for a in frame_to_applicants(X, card.features)]
        )
        np.testing.assert_allclose(from_json, card.predict_proba_bad(X), atol=1e-9)

    def test_missing_values_agree(self, fitted):
        """The NaN path is the easiest one to get wrong in a reimplementation."""
        card, X, _ = fitted
        has_nan = X[card.features].isna().any(axis=1)
        assert has_nan.sum() > 0, "fixture should contain missing values"

        subset = X[has_nan]
        model = ScoringModel(to_dict(card))
        from_json = np.array(
            [model.score(a) for a in frame_to_applicants(subset, card.features)]
        )
        np.testing.assert_allclose(from_json, card.score(subset).to_numpy(), atol=1e-9)

    def test_bin_edges_are_right_closed(self, fitted):
        """A value sitting exactly on an edge belongs to the bin below it.

        pd.cut is right-closed, so the serving side has to be too, or applicants
        on a boundary get priced from the wrong bin.
        """
        card, _, _ = fitted
        model = ScoringModel(to_dict(card))
        bins = model.features["age"]["bins"]
        edge = bins[0]["right"]

        assert model.points_for("age", edge) == pytest.approx(bins[0]["points"])
        assert model.points_for("age", edge + 1e-9) == pytest.approx(bins[1]["points"])


class TestExportShape:
    def test_is_valid_strict_json(self, fitted, tmp_path):
        """No Infinity, no NaN. json.loads with strict parsing has to accept it."""
        card, _, _ = fitted
        path = export(card, tmp_path / "scorecard.json")
        text = path.read_text(encoding="utf-8")
        assert "Infinity" not in text and "NaN" not in text
        json.loads(text, parse_constant=_reject)

    def test_unbounded_edges_become_null(self, fitted):
        card, _, _ = fitted
        spec = to_dict(card)
        age = next(f for f in spec["features"] if f["name"] == "age")
        assert age["bins"][0]["left"] is None
        assert age["bins"][-1]["right"] is None

    def test_every_model_feature_is_present(self, fitted):
        card, _, _ = fitted
        spec = to_dict(card)
        assert [f["name"] for f in spec["features"]] == list(card.features)

    def test_round_trips_through_a_file(self, fitted, tmp_path):
        card, X, _ = fitted
        path = export(card, tmp_path / "scorecard.json")
        model = ScoringModel.from_json(path)
        applicant = frame_to_applicants(X.head(1), card.features)[0]
        assert model.score(applicant) == pytest.approx(card.score(X.head(1)).iloc[0])

    def test_rejects_an_unknown_schema_version(self, fitted):
        card, _, _ = fitted
        spec = to_dict(card)
        spec["schema_version"] = SCHEMA_VERSION + 99
        with pytest.raises(ValueError, match="schema_version"):
            ScoringModel(spec)


class TestScoringBehaviour:
    def test_unseen_counter_value_scores_neutral(self, fitted):
        """A value the training data never held contributes nothing either way."""
        card, _, _ = fitted
        model = ScoringModel(to_dict(card))
        assert model.points_for("late_code_flag", 7.0) == 0.0

    def test_counter_is_capped_the_same_way(self, fitted):
        """Above the cap, every value is the same risk statement."""
        card, _, _ = fitted
        model = ScoringModel(to_dict(card))
        cap = cfg.CLIP_AT["NumberOfTimes90DaysLate"]
        assert model.points_for("NumberOfTimes90DaysLate", cap) == pytest.approx(
            model.points_for("NumberOfTimes90DaysLate", cap + 10)
        )

    def test_missing_field_is_treated_as_missing(self, fitted):
        """A key absent from the payload must behave like an explicit null."""
        card, _, _ = fitted
        model = ScoringModel(to_dict(card))
        absent = {"NumberOfTimes90DaysLate": 0.0, "late_code_flag": 0}
        explicit = {**absent, "age": None}
        assert model.score(absent) == pytest.approx(model.score(explicit))

    def test_breakdown_sums_to_the_score(self, fitted):
        card, X, _ = fitted
        model = ScoringModel(to_dict(card))
        applicant = frame_to_applicants(X.head(1), card.features)[0]
        assert model.base_points + sum(model.breakdown(applicant).values()) == pytest.approx(
            model.score(applicant)
        )

    def test_reasons_are_ordered_by_points_lost(self, fitted):
        card, _, _ = fitted
        model = ScoringModel(to_dict(card))
        worst = {"age": 22.0, "NumberOfTimes90DaysLate": 5.0, "late_code_flag": 1}
        reasons = model.reasons(worst)
        lost = [r["points_lost"] for r in reasons]
        assert lost == sorted(lost, reverse=True)
        assert all(v > 0 for v in lost)

    def test_the_best_applicant_has_no_reasons(self, fitted):
        card, _, _ = fitted
        model = ScoringModel(to_dict(card))
        best = {}
        for name in model.feature_names:
            spec = model.features[name]
            # one value landing in each bin, plus None for the missing bin
            candidates = [None]
            for b in spec["bins"]:
                if "value" in b:
                    candidates.append(b["value"])
                elif b["right"] is not None:
                    candidates.append(b["right"])       # right-closed, so this lands here
                else:
                    candidates.append(b["left"] + 1.0)
            best[name] = max(candidates, key=lambda v: model.points_for(name, v))

        assert model.reasons(best) == []

    def test_assess_returns_a_tier_and_a_probability(self, fitted):
        card, X, _ = fitted
        model = ScoringModel(to_dict(card))
        out = model.assess(frame_to_applicants(X.head(1), card.features)[0])
        assert out["tier"] in {"A", "B", "C"}
        assert 0.0 < out["probability_of_default"] < 1.0
        assert set(out["points"]) == set(model.feature_names)


def _reject(constant):
    raise ValueError(f"non-JSON constant in export: {constant}")
