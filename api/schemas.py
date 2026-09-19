"""Request and response shapes for the scoring API.

Validation bounds are not decoration. The model was fit on a population, and
outside that population its bins say nothing useful, so the service refuses
rather than returning a confident number it cannot support. Every bound below
either comes from a physical fact (age, counts) or from what the training data
actually held.

Optional fields are optional on purpose. Missingness is a modelled bin with its
own WoE, not an error, so an application with no income figure scores as the
group that has no income figure rather than being rejected.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

# Bounds in one place so the API and the UI cannot disagree about them.
AGE_MIN, AGE_MAX = 18, 110
UTIL_MAX = 13.0          # above this the training data treats the value as broken
COUNT_MAX = 20           # the real counters top out at 17; 20 leaves headroom
INCOME_MAX = 1_000_000.0
DEBT_RATIO_MAX = 100.0


class Applicant(BaseModel):
    """One credit application, in the units the source data uses."""

    model_config = {
        "extra": "forbid",
        "json_schema_extra": {
            "examples": [
                {
                    "RevolvingUtilizationOfUnsecuredLines": 0.25,
                    "age": 45,
                    "NumberOfTime30-59DaysPastDueNotWorse": 0,
                    "DebtRatio": 0.35,
                    "MonthlyIncome": 5400,
                    "NumberOfOpenCreditLinesAndLoans": 8,
                    "NumberOfTimes90DaysLate": 0,
                    "NumberRealEstateLoansOrLines": 1,
                    "NumberOfTime60-89DaysPastDueNotWorse": 0,
                    "NumberOfDependents": 2,
                }
            ]
        },
    }

    revolving_utilization: Annotated[float, Field(ge=0, le=UTIL_MAX, alias="RevolvingUtilizationOfUnsecuredLines")]
    age: Annotated[int, Field(ge=AGE_MIN, le=AGE_MAX)]
    late_30_59: Annotated[int, Field(ge=0, le=COUNT_MAX, alias="NumberOfTime30-59DaysPastDueNotWorse")]
    debt_ratio: Annotated[float, Field(ge=0, le=DEBT_RATIO_MAX, alias="DebtRatio")]
    monthly_income: Annotated[float | None, Field(default=None, ge=0, le=INCOME_MAX, alias="MonthlyIncome")]
    open_credit_lines: Annotated[int, Field(ge=0, le=100, alias="NumberOfOpenCreditLinesAndLoans")]
    late_90: Annotated[int, Field(ge=0, le=COUNT_MAX, alias="NumberOfTimes90DaysLate")]
    real_estate_loans: Annotated[int, Field(ge=0, le=60, alias="NumberRealEstateLoansOrLines")]
    late_60_89: Annotated[int, Field(ge=0, le=COUNT_MAX, alias="NumberOfTime60-89DaysPastDueNotWorse")]
    dependents: Annotated[int | None, Field(default=None, ge=0, le=25, alias="NumberOfDependents")]

    # No cross-field validator here on purpose. The three delinquency counters
    # count distinct events in a window, so a borrower can hold a 90-day late
    # with no 30-59 day late recorded, and the training data contains exactly
    # that. Inventing a consistency rule would reject real applications.

    def to_features(self) -> dict[str, float | None]:
        """Reproduce the derived fields the model was trained on.

        Only `debt_ratio`, the ratio branch of the DebtRatio split, which the
        model reads when income is present and treats as missing when it is not.

        No late_code_flag: it is collinear with the counters' missing bins and is
        not in MODEL_FEATURES. See the note in config.py.
        """
        income_known = self.monthly_income is not None
        return {
            "RevolvingUtilizationOfUnsecuredLines": self.revolving_utilization,
            "age": float(self.age),
            "NumberOfTime30-59DaysPastDueNotWorse": float(self.late_30_59),
            "NumberOfTimes90DaysLate": float(self.late_90),
            "NumberOfTime60-89DaysPastDueNotWorse": float(self.late_60_89),
            "MonthlyIncome": self.monthly_income,
            "debt_ratio": self.debt_ratio if income_known else None,
            "NumberOfOpenCreditLinesAndLoans": float(self.open_credit_lines),
            "NumberRealEstateLoansOrLines": float(self.real_estate_loans),
            "NumberOfDependents": None if self.dependents is None else float(self.dependents),
        }


class Reason(BaseModel):
    feature: str
    points: float
    points_lost: float


class Decision(BaseModel):
    score: float
    probability_of_default: float
    tier: str
    decision: str
    base_points: float
    points: dict[str, float]
    reasons: list[Reason]
    cutoff: float


class Health(BaseModel):
    status: str
    model_loaded: bool
    schema_version: int
    features: int
    cutoff: float
