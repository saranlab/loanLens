"""Thresholds, bin edges and scorecard parameters.

Every number a reviewer might want to argue with lives here rather than inline
in the pipeline. Section references point at `notebook/exploratory_data_analysis.ipynb`,
which is where each value was derived.
"""

from __future__ import annotations

import numpy as np

TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# The three delinquency counters. Grouped because the 96/98 status codes always
# appear in all three at once (section 5.1).
LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

# Bureau status codes, not counts. 269 rows, bad rate 54.7% (section 5.1).
SENTINEL_CODES = (96, 98)

# Above this, utilization has a 5.9% bad rate, below the 6.7% portfolio average,
# so the values contradict the risk they would imply. Utilization between 1 and 2
# stays: 40.1% bad rate is real risk (section 5.2).
UTIL_INVALID_ABOVE = 13.0

MIN_AGE = 18

# Bin edges from the bad-rate-per-bin analysis in section 7. Chosen so that each
# bin has a reason (0.9 and 1.0 bracket the credit limit; the age bands are
# five-year steps where the bad rate actually moves) and no bin is too thin to
# estimate. Values outside the outer edges cannot occur because the edges are
# unbounded, which keeps transform() safe on unseen data.
BIN_EDGES: dict[str, list[float]] = {
    "RevolvingUtilizationOfUnsecuredLines": [-np.inf, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 2.0, np.inf],
    "age": [-np.inf, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, np.inf],
    "debt_ratio": [-np.inf, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, np.inf],
    "MonthlyIncome": [-np.inf, 1500, 3000, 4500, 6000, 8000, 11000, np.inf],
    "NumberOfOpenCreditLinesAndLoans": [-np.inf, 2, 4, 6, 8, 11, 15, np.inf],
}

# Discrete counters are capped rather than binned by edges: the tail is thin and
# the bad rate is flat past the cap, so "3 or more" is the same risk statement as
# "exactly 7". Each remaining integer becomes its own bin.
CLIP_AT: dict[str, int] = {
    "NumberOfTimes90DaysLate": 3,
    "NumberOfTime30-59DaysPastDueNotWorse": 4,
    "NumberOfTime60-89DaysPastDueNotWorse": 2,
    "NumberRealEstateLoansOrLines": 4,
    "NumberOfDependents": 4,
}

# Already 0/1, so they pass through binning as two bins.
FLAG_COLS = ["late_code_flag"]

# What goes into the scorecard.
#
# weighted_late (IV 1.51) beats NumberOfTimes90DaysLate (0.88) and is excluded
# anyway: a scorecard row has to be explainable to a declined applicant, and
# "weighted delinquency index" is not. The three raw counters each get their own
# row instead. Same reasoning drops disposable_income (IV 0.126 vs 0.074 for
# MonthlyIncome), which is collinear with debt_ratio and MonthlyIncome together.
#
# Both belong in a tree model. See section 10 for the IV comparison.
MODEL_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "NumberOfTimes90DaysLate",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "age",
    "NumberOfOpenCreditLinesAndLoans",
    "MonthlyIncome",
    "debt_ratio",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
    "late_code_flag",
]

# Haldane correction, so a bin with no bads gives a finite WoE instead of -inf.
WOE_EPS = 0.5

# Drop a feature below this IV. 0.02 is the conventional floor for "carries
# anything at all".
MIN_IV = 0.02

# Scorecard scaling (section 2 of the plan): 600 points at 50:1 good:bad odds,
# and every 20 points doubles the odds.
BASE_SCORE = 600.0
BASE_ODDS = 50.0
PDO = 20.0

# Score cutoffs for the risk tiers the API reports.
TIER_CUTOFFS = [(720, "A"), (620, "B")]
TIER_FLOOR = "C"
