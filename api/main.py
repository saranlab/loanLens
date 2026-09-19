"""Scoring service.

    uvicorn api.main:app --reload

Reads `artifacts/scorecard.json` at startup and holds it in memory. No sklearn,
no pandas, no pickle: the model here is a lookup table, so the container needs
nothing from the training stack.

The service returns the reasons alongside the decision, because a declined
applicant is owed them and retrofitting explanations onto a scoring endpoint
later is how they end up being written by hand and drifting from the model.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from api.schemas import Applicant, Decision, Health
from src.scoring import ScoringModel

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.environ.get("SCORECARD_PATH", ROOT / "artifacts" / "scorecard.json"))

# The approval line, from src/policy.py: the profit optimum at the assumed 8%
# margin and 60% LGD. Overridable because it is a business input, not a model
# output, and it moves with the loss ratio.
CUTOFF = float(os.environ.get("APPROVAL_CUTOFF", 550.0))

state: dict[str, ScoringModel | None] = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the scorecard once, at startup.

    Loading per request would hide a missing or corrupt artifact until the first
    applicant arrives. Failing at startup means a bad deploy never takes traffic.
    """
    if MODEL_PATH.exists():
        state["model"] = ScoringModel.from_json(MODEL_PATH)
    else:
        state["model"] = None
    yield
    state["model"] = None


app = FastAPI(
    title="LoanLens scoring",
    version="1.0.0",
    summary="Credit scorecard with adverse action reasons",
    lifespan=lifespan,
)


def get_model() -> ScoringModel:
    model = state["model"]
    if model is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"no scorecard at {MODEL_PATH}. Run `python -m src.train` first.",
        )
    return model


@app.get("/health", response_model=Health)
def health() -> Health:
    """Readiness, including whether the artifact actually loaded."""
    model = state["model"]
    return Health(
        status="ok" if model else "no model",
        model_loaded=model is not None,
        schema_version=model.spec["schema_version"] if model else 0,
        features=len(model.feature_names) if model else 0,
        cutoff=CUTOFF,
    )


@app.post("/score", response_model=Decision)
def score(applicant: Applicant) -> Decision:
    """Score one application and say why."""
    model = get_model()
    assessment = model.assess(applicant.to_features())
    return Decision(
        **assessment,
        decision="approve" if assessment["score"] >= CUTOFF else "decline",
        cutoff=CUTOFF,
    )


@app.post("/score/batch", response_model=list[Decision])
def score_batch(applicants: list[Applicant]) -> list[Decision]:
    if len(applicants) > 1000:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            detail="at most 1000 applications per request",
        )
    return [score(a) for a in applicants]


@app.get("/scorecard")
def scorecard() -> JSONResponse:
    """The auditable points table.

    Published to provide complete regulatory explainability and algorithmic
    transparency under fair lending standards (FCRA, ECOA).
    """
    model = get_model()
    return JSONResponse(
        {
            "schema_version": model.spec["schema_version"],
            "scaling": model.scaling,
            "base_points": round(model.base_points, 2),
            "tiers": model.tiers,
            "cutoff": CUTOFF,
            "features": [
                {
                    "name": f["name"],
                    "iv": round(f["iv"], 4),
                    "kind": f["kind"],
                    "bins": [
                        {**{k: v for k, v in b.items() if k != "woe"},
                         "points": round(b["points"], 2)}
                        for b in f["bins"]
                    ],
                    "missing_points": round(f["missing"]["points"], 2),
                }
                for f in model.spec["features"]
            ],
        }
    )
