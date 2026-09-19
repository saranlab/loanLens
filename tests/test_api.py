"""Tests for the scoring service.

Builds its own scorecard into a temp file rather than reading the real artifact,
so the suite still runs on a clone with no Kaggle data and no trained model.
"""

from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src import config as cfg
from src.binning import WoEBinner
from src.scorecard import Scorecard
from src.serving import export

SAFE = {
    "RevolvingUtilizationOfUnsecuredLines": 0.05,
    "age": 62,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.2,
    "MonthlyIncome": 12000,
    "NumberOfOpenCreditLinesAndLoans": 8,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 0,
}

RISKY = {
    **SAFE,
    "RevolvingUtilizationOfUnsecuredLines": 1.4,
    "age": 24,
    "NumberOfTime30-59DaysPastDueNotWorse": 3,
    "MonthlyIncome": 1800,
    "NumberOfTimes90DaysLate": 3,
    "NumberOfTime60-89DaysPastDueNotWorse": 2,
}


def _train_a_scorecard(path):
    """A small but real scorecard over the features the API sends."""
    rng = np.random.default_rng(5)
    n = 12_000
    X = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": rng.random(n) * 1.5,
            "age": rng.integers(21, 85, n).astype(float),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.integers(0, 5, n).astype(float),
            "NumberOfTimes90DaysLate": rng.integers(0, 4, n).astype(float),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.integers(0, 3, n).astype(float),
            "MonthlyIncome": rng.lognormal(8.5, 0.6, n),
            "debt_ratio": rng.random(n) * 1.5,
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 20, n).astype(float),
            "NumberRealEstateLoansOrLines": rng.integers(0, 5, n).astype(float),
            "NumberOfDependents": rng.integers(0, 5, n).astype(float),
        }
    )
    risk = (
        0.01
        + 0.25 * (X["RevolvingUtilizationOfUnsecuredLines"] > 0.9)
        + 0.20 * (X["NumberOfTimes90DaysLate"] > 0)
        + 0.10 * (X["NumberOfTime30-59DaysPastDueNotWorse"] > 1)
        + 0.06 * (X["age"] < 35)
    )
    y = pd.Series((rng.random(n) < risk.clip(0, 0.95)).astype(int), index=X.index)

    binner = WoEBinner(cfg.MODEL_FEATURES, min_iv=0.0).fit(X, y)
    model = LogisticRegression(max_iter=1000).fit(binner.transform(X), y)
    export(Scorecard(binner, model), path)


@pytest.fixture
def client(tmp_path, monkeypatch):
    spec = tmp_path / "scorecard.json"
    _train_a_scorecard(spec)
    monkeypatch.setenv("SCORECARD_PATH", str(spec))
    monkeypatch.setenv("APPROVAL_CUTOFF", "550")

    import api.main

    importlib.reload(api.main)
    # the context manager is what runs lifespan, which is what loads the model
    with TestClient(api.main.app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("SCORECARD_PATH", str(tmp_path / "absent.json"))
    import api.main

    importlib.reload(api.main)
    with TestClient(api.main.app) as c:
        yield c


class TestHealth:
    def test_reports_a_loaded_model(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["features"] == len(cfg.MODEL_FEATURES)

    def test_reports_a_missing_model_rather_than_pretending(self, client_without_model):
        body = client_without_model.get("/health").json()
        assert body["model_loaded"] is False
        assert body["status"] != "ok"

    def test_scoring_without_a_model_is_503_not_500(self, client_without_model):
        """An operational problem should read as one, so a probe can act on it."""
        r = client_without_model.post("/score", json=SAFE)
        assert r.status_code == 503
        assert "train" in r.json()["detail"]


class TestScoring:
    def test_safe_applicant_is_approved(self, client):
        body = client.post("/score", json=SAFE).json()
        assert body["decision"] == "approve"
        assert body["score"] >= body["cutoff"]

    def test_risky_applicant_is_declined(self, client):
        body = client.post("/score", json=RISKY).json()
        assert body["decision"] == "decline"
        assert body["score"] < body["cutoff"]

    def test_safe_scores_above_risky(self, client):
        safe = client.post("/score", json=SAFE).json()
        risky = client.post("/score", json=RISKY).json()
        assert safe["score"] > risky["score"]
        assert safe["probability_of_default"] < risky["probability_of_default"]

    def test_points_sum_to_the_score(self, client):
        """The response has to reconcile, or the explanation is decoration."""
        body = client.post("/score", json=SAFE).json()
        total = body["base_points"] + sum(body["points"].values())
        assert total == pytest.approx(body["score"], abs=0.05)

    def test_every_model_feature_gets_a_line(self, client):
        body = client.post("/score", json=SAFE).json()
        assert set(body["points"]) == set(cfg.MODEL_FEATURES)

    def test_declines_come_with_reasons(self, client):
        body = client.post("/score", json=RISKY).json()
        assert body["reasons"], "a declined applicant is owed reasons"
        lost = [r["points_lost"] for r in body["reasons"]]
        assert lost == sorted(lost, reverse=True)

    def test_no_feature_is_a_reason_for_everyone(self, client):
        """Guards the late_code_flag failure: a constant deduction is not a reason.

        If a feature appears in every applicant's reasons, it is being charged to
        people who did nothing, and it says nothing about the individual.
        """
        bodies = [client.post("/score", json=p).json() for p in (SAFE, RISKY)]
        safe_reasons = {r["feature"] for r in bodies[0]["reasons"]}
        risky_reasons = {r["feature"] for r in bodies[1]["reasons"]}
        assert safe_reasons != risky_reasons

    def test_tier_is_one_of_the_published_ones(self, client):
        body = client.post("/score", json=SAFE).json()
        assert body["tier"] in {"A", "B", "C"}


class TestMissingData:
    def test_income_may_be_omitted(self, client):
        """Missingness is a modelled bin, so it is an answer rather than an error."""
        payload = {k: v for k, v in SAFE.items() if k != "MonthlyIncome"}
        r = client.post("/score", json=payload)
        assert r.status_code == 200
        assert r.json()["points"]["MonthlyIncome"] is not None

    def test_explicit_null_matches_an_absent_key(self, client):
        absent = {k: v for k, v in SAFE.items() if k != "NumberOfDependents"}
        explicit = {**absent, "NumberOfDependents": None}
        assert (
            client.post("/score", json=absent).json()["score"]
            == client.post("/score", json=explicit).json()["score"]
        )


class TestValidation:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("age", 4),
            ("age", 200),
            ("RevolvingUtilizationOfUnsecuredLines", -0.1),
            ("RevolvingUtilizationOfUnsecuredLines", 50_000),
            ("MonthlyIncome", -1),
            ("NumberOfTimes90DaysLate", -1),
            ("NumberOfTimes90DaysLate", 98),
            ("DebtRatio", -0.5),
        ],
    )
    def test_out_of_range_is_rejected(self, client, field, value):
        r = client.post("/score", json={**SAFE, field: value})
        assert r.status_code == 422

    def test_utilization_above_the_valid_range_is_rejected(self, client):
        """13 is where the training data stops treating the value as real."""
        assert client.post(
            "/score", json={**SAFE, "RevolvingUtilizationOfUnsecuredLines": 13.1}
        ).status_code == 422

    def test_unknown_fields_are_rejected(self, client):
        """A typo in a field name must not silently score the default instead."""
        r = client.post("/score", json={**SAFE, "MontlyIncome": 5000})
        assert r.status_code == 422

    def test_missing_required_field_is_rejected(self, client):
        payload = {k: v for k, v in SAFE.items() if k != "age"}
        assert client.post("/score", json=payload).status_code == 422


class TestBatch:
    def test_scores_each_application(self, client):
        body = client.post("/score/batch", json=[SAFE, RISKY]).json()
        assert len(body) == 2
        assert body[0]["score"] > body[1]["score"]

    def test_rejects_an_oversized_batch(self, client):
        assert client.post("/score/batch", json=[SAFE] * 1001).status_code == 413


class TestScorecardEndpoint:
    def test_publishes_the_points_table(self, client):
        body = client.get("/scorecard").json()
        assert len(body["features"]) == len(cfg.MODEL_FEATURES)
        assert body["cutoff"] == 550.0
        for feature in body["features"]:
            assert feature["bins"], feature["name"]
            assert all("points" in b for b in feature["bins"])

    def test_does_not_leak_raw_woe(self, client):
        """Points are the unit a reviewer works in; WoE is an implementation detail."""
        body = client.get("/scorecard").json()
        assert "woe" not in json.dumps(body)


class TestOpenAPI:
    def test_schema_is_generated(self, client):
        schema = client.get("/openapi.json").json()
        assert "/score" in schema["paths"]
        assert "Applicant" in schema["components"]["schemas"]
