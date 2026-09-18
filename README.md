# LoanLens

A credit scorecard built on the *Give Me Some Credit* dataset: binned features,
WoE transformation, logistic regression, and a PDO-scaled score, with the
reasoning for each decision written down rather than assumed.

Target is `SeriousDlqin2yrs`, whether a borrower went 90+ days delinquent within
two years. 150,000 training rows, 6.68% bad rate.

## Status

- [x] EDA and data quality analysis (`notebook/exploratory_data_analysis.ipynb`)
- [x] Preprocessing module extracted from the notebook
- [x] Binning, WoE, logistic regression, PDO scorecard
- [ ] FastAPI scoring service
- [ ] Streamlit reviewer interface
- [ ] Docker Compose

## Getting the data

The CSVs are not in the repo. Download them from
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place
them like this:

```
data/raw/cs-training.csv
data/test/cs-test.csv
```

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows; use .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Read the analysis:

```bash
jupyter lab notebook/exploratory_data_analysis.ipynb
```

Fit the scorecard and write `artifacts/scorecard.joblib`:

```bash
python -m src.train
pytest -q
```

## Results

Held-out test set, 30,018 rows:

| Metric | Test | Train | 5-fold CV |
|---|---|---|---|
| AUC | 0.8556 | 0.8596 | 0.8591 +/- 0.0036 |
| Gini | 0.7111 | 0.7193 | |
| KS | 0.5583 | 0.5612 | |

The train/test gap is 0.004 and the CV spread is 0.0036, so the holdout figure is
not a lucky split.

Risk tiers are stated as the default probability they stand for, and the score
cutoffs are derived from the PDO scale rather than picked. On the test set they
hold up:

| Tier | Promise | Cutoff | Share | Observed bad rate |
|---|---|---|---|---|
| A | under 1% PD | 620 | 5.5% | 0.48% |
| B | under 5% PD | 572 | 64.4% | 1.87% |
| C | the rest | | 30.1% | 18.11% |

Scaling is 600 points at 50:1 good:bad odds with 20 points per doubling, so 620
is 100:1 and 640 is 200:1. The original plan put tier A at 720, which on this
scale needs a PD of 0.03%; no applicant in this portfolio is that safe, so the
tier would have been empty.

## What the EDA found

Four things that change how the data has to be handled:

**The delinquency counters use status codes.** 269 rows hold 96 or 98 in all
three counters at once, with nothing between 13 and 96. These are bureau status
codes, not counts. Their bad rate is 54.7%, eight times the overall rate, so the
value becomes NaN and the signal is kept as a flag.

**`DebtRatio` holds two different units.** Of the 28,877 rows above 10, 26,771
(92.7%) also have no income. With no denominator to divide by, the source system
recorded a monthly amount instead of a ratio. The column is split in two,
switched by an income-missing flag.

**Capping utilization at 1.0 would be wrong.** Utilization between 1 and 2 has a
40.1% bad rate against 19.4% just below 1.0, so exceeding a credit limit is real
risk rather than a data error. Values above 13 are the errors: their bad rate is
5.9%, below the portfolio average.

**Four features have IV above 0.5.** The usual reading is to suspect leakage, but
payment history and utilization are bureau facts known at application time.
Dropping all four would leave `age` at IV 0.26 as the strongest remaining
feature. `NumberOfTimes90DaysLate` is the one open question: its definition
states no time window while the other two counters state "in the last 2 years".

## Layout

```
data/raw/          cs-training.csv (gitignored)
data/test/         cs-test.csv (gitignored)
notebook/          exploratory data analysis
src/config.py      thresholds, bin edges, scorecard parameters
src/preprocessing  cleaning and derived features, stateless
src/binning.py     WoE encoding, fit on training rows only
src/scorecard.py   coefficients to points, adverse action reasons
src/evaluate.py    AUC, KS, Gini, calibration
src/train.py       entry point
tests/             68 tests, no dataset required
artifacts/         serialized model (gitignored)
```

Everything that has to be estimated from data lives in `binning.py` and is fit
inside the training fold. `preprocessing.py` is stateless by design, with tests
asserting that cleaning a subset matches cleaning everything and then subsetting,
which is what makes it safe to run before the split.
