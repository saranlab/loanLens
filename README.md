# LoanLens

A credit scorecard built on the *Give Me Some Credit* dataset: binned features,
WoE transformation, logistic regression, and a PDO-scaled score, with the
reasoning for each decision written down rather than assumed.

Target is `SeriousDlqin2yrs`, whether a borrower went 90+ days delinquent within
two years. 150,000 training rows, 6.68% bad rate.

## Status

- [x] EDA and data quality analysis (`notebook/exploratory_data_analysis.ipynb`)
- [ ] Preprocessing module extracted from the notebook
- [ ] Binning, WoE, logistic regression, PDO scorecard
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
jupyter lab notebook/exploratory_data_analysis.ipynb
```

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
src/               preprocessing and model code
artifacts/         serialized model (gitignored)
```
