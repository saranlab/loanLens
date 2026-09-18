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

# Bin edges from the bad-rate-per-bin analysis in section 7. Each edge has a
# reason: 0.9 and 1.0 bracket the credit limit, the age bands are five-year steps
# over the range where the bad rate actually moves. The outer edges are unbounded,
# so no value can fall outside the bins and transform() stays safe on data it has
# not seen. Where the source analysis split a tail too finely, the bins are merged
# here rather than left to break monotonicity on a handful of rows.
BIN_EDGES: dict[str, list[float]] = {
    # Everything over the credit limit is one bin. Splitting at 2.0 left only 105
    # training rows above it, and that bin's bad rate came out *below* the bin
    # beneath it, which broke monotonicity on noise rather than on signal.
    "RevolvingUtilizationOfUnsecuredLines": [-np.inf, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0, np.inf],
    # Under 30 is one bin. The bad rate peaks at 25-30 (11.35%) rather than at the
    # youngest band (11.00%), so splitting there is not monotonic. Merging is
    # honest about it and "under 30" is a band a credit officer already thinks in.
    "age": [-np.inf, 30, 35, 40, 45, 50, 55, 60, 65, 70, np.inf],
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

# A bin below this many training rows is scored at the portfolio average instead
# of getting its own estimate. At a 6.7% base rate, 100 rows hold about 7 bads,
# which is already a shaky rate estimate; below that the number is noise. The
# usual scorecard convention asks for 5% of the population per bin, which the
# tail of a strong feature cannot deliver, so this is a floor rather than a
# target. Thin bins are reported by WoEBinner.thin_bins_ either way.
MIN_BIN_COUNT = 100

# Drop a feature below this IV. 0.02 is the conventional floor for "carries
# anything at all".
MIN_IV = 0.02

# Scorecard scaling (section 2 of the plan): 600 points at 50:1 good:bad odds,
# and every 20 points doubles the odds.
BASE_SCORE = 600.0
BASE_ODDS = 50.0
PDO = 20.0

# Risk tiers, defined by the default probability they stand for rather than by a
# round number of points. The plan this project started from proposed A at 720
# and B at 620, which came from the FICO range, not from this scale: at 600 = 50:1
# and PDO 20, reaching 720 needs odds of 50 * 2**6 = 3200:1, a PD of 0.03%. No
# applicant in this portfolio is that safe, so tier A would have been empty.
#
# Stating the PD boundary instead makes the tier mean something, and the cutoff
# points fall out of the scaling formula. src.scorecard.tier_cutoffs() does the
# conversion so the two can never drift apart.
TIER_MAX_PD = [(0.01, "A"), (0.05, "B")]
TIER_FLOOR = "C"
