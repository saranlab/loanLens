# LoanLens

[![tests](https://github.com/saranlab/loanLens/actions/workflows/tests.yml/badge.svg)](https://github.com/saranlab/loanLens/actions/workflows/tests.yml)

A credit scorecard on the *Give Me Some Credit* dataset: WoE binning, logistic
regression, PDO-scaled points, and adverse action reasons. Target is
`SeriousDlqin2yrs`, whether a borrower went 90+ days delinquent within two years.
150,000 rows, 6.68% bad rate.

The dataset is well worn. What is less common is that **four steps of the standard
recipe for it turn out to be wrong**, and this repo shows the number behind each
one rather than following the recipe.

| Standard advice | What the data says |
|---|---|
| Cap utilization above 1.0, since a ratio cannot exceed 1 | Utilization between 1 and 2 has a **40.1%** bad rate against **19.4%** just below 1.0. Capping there prices 2,950 borrowers at half their real risk. The actual errors sit above 13: **5.9%** bad, *below* the 6.7% portfolio average |
| Impute missing income with the median | Missing income carries a **lower** bad rate than present income (5.61% vs 6.95%), so the gap is itself a signal. WoE binning gives it its own bin, and no imputation is needed anywhere in the pipeline |
| Treat IV above 0.5 as leakage and drop it | Four features exceed 0.5. Dropping them leaves `age` at IV 0.26 as the strongest thing remaining. Bureau history is known at application time, so it is not leakage. The test is a question about timing, not a threshold |
| Put tier A at 720 points | On a 600 = 50:1 scale with PDO 20, 720 needs a PD of 0.03%. The safest applicant here scores 630, so tier A would be empty. Tiers are defined by the PD they stand for and the cutoffs derived from the scale |

Two things the dataset documentation does not mention:

**The delinquency counters carry status codes.** 269 rows hold 96 or 98 in all
three counters at once, with nothing at all between 13 and 96. Their bad rate is
**54.7%**, eight times the average, which makes the code itself the strongest
single signal in the data. Left as counts, a model reads them as 98 late payments.

**`DebtRatio` holds two different units.** Of the 28,877 rows above 10, 26,771
(**92.7%**) also have no income. With no denominator available the source system
recorded a monthly amount instead of a ratio, so the column is split in two rather
than capped.

## Results

Held-out test set of 30,018 rows, fit on the other 119,982:

| Metric | Test | Train | 5-fold CV |
|---|---|---|---|
| AUC | **0.8556** | 0.8596 | 0.8591 +/- 0.0036 |
| Gini | 0.7111 | 0.7193 | |
| KS | 0.5583 | 0.5612 | |

The train/test gap is 0.004 and the CV spread 0.0036, so the holdout figure is not
a lucky split.

Risk tiers state the default probability they stand for, and their score cutoffs
come out of the PDO formula. On the test set they hold:

| Tier | Promise | Cutoff | Share | Observed bad rate |
|---|---|---|---|---|
| A | under 1% PD | 620 | 5.5% | 0.48% |
| B | under 5% PD | 572 | 64.4% | 1.87% |
| C | the rest | | 30.1% | 18.11% |

Scaling is 600 points at 50:1 good:bad odds with 20 points per doubling, so 620
reads as 100:1 and 640 as 200:1.

### What staying interpretable costs

LightGBM on the same split, given raw values and the features the scorecard drops
for explainability:

| Model | AUC | Gini | KS |
|---|---|---|---|
| LightGBM | 0.8636 | 0.7271 | 0.5803 |
| Scorecard | 0.8556 | 0.7111 | 0.5583 |
| **Gap** | **0.0080** | 0.0160 | 0.0220 |

0.94% of AUC, against a seed-to-seed spread of 0.0004 for the tree itself, so the
gap is real and small. Where it comes from is the interesting part: `weighted_late`
takes 39.8% of the tree's gain and `total_late` another 16.1%, which are precisely
the two features `config.py` excludes because "weighted delinquency index" is not
something a declined applicant can be told. The 0.0080 is the price of that
decision, and it is now quoted rather than guessed at.

### Where to set the approval cutoff

AUC ranks applicants but says nothing about where to draw the line. Approving pays
when `(1 - p) * margin > p * loss`, so the threshold is `p < margin / (margin + loss)`.

At an assumed 8% margin and 60% loss given default, that break-even PD is 11.76%,
a cutoff of 545. Sweeping the test set puts the optimum at 550: approve 85.5% with
2.9% bad among those approved.

| Cutoff | Approval rate | Bad rate approved | Profit per application |
|---|---|---|---|
| 500 | 96.5% | 5.03% | 442 |
| 540 | 89.7% | 3.42% | 509 |
| **550** | **85.5%** | **2.91%** | **515** |
| 572 (tier B floor) | 69.9% | 1.76% | 475 |
| 600 | 39.4% | 0.91% | 291 |

Using the tier B floor as the approval line costs 39.5 per application, 7.7%. A
tier states a risk level; a cutoff trades margin against loss. Different questions.

Every figure there rests on the margin and loss assumptions, so `src/policy.py`
sweeps the loss-to-margin ratio from 2.5 to 25 and the optimal cutoff moves from
516 to 584. The honest answer to "where should the cutoff be" is that it moves
with your loss ratio.

## Running it

The CSVs are not in the repo. Download them from
[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) into
`data/raw/cs-training.csv` and `data/test/cs-test.csv`.

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows; .venv/bin/activate elsewhere
pip install -r requirements.txt

python -m src.train      # fits the scorecard, writes artifacts/scorecard.joblib
python -m src.baseline   # LightGBM comparison
python -m src.policy     # approval cutoff economics
pytest -q                # 84 tests, no dataset needed
```

The analysis is in `notebook/exploratory_data_analysis.ipynb`, committed with its
outputs so it reads on GitHub without the data.

## How leakage is kept out

The split comes first. Everything fitted afterwards sees training rows only, and
the test set is touched once, by `transform`, to be scored.

`preprocessing.py` is stateless on purpose: no row's output depends on any other
row, so it is safe to run before the split. Two tests assert that by cleaning a
subset and comparing against cleaning everything and then subsetting.

Anything estimated from data lives in `binning.py`. Bin membership comes from fixed
edges, but the WoE attached to each bin is computed from the target, which is what
makes the ordering matter. A test encodes the same test rows twice, once with a
binner fit on train alone and once with one fit on the pool, and asserts the
results differ.

Cross-validation refits the binner inside each fold rather than reusing one fit
from outside the loop.

## Two routes to the same score

`Scorecard.score()` goes through the model. `Scorecard.score_from_points()` sums
the points table a human would read. `python -m src.train` fails if they disagree.

In production the points table often *is* the model: a credit officer or a core
banking system adds up points from a table, not a pickled estimator. If the two
drift apart, the validated model is not the thing deciding, and the reasons given
for a decline are fiction.

## Layout

```
notebook/            exploratory data analysis, with outputs
src/config.py        thresholds, bin edges, scorecard parameters, all in one place
src/preprocessing.py cleaning and derived features, stateless
src/binning.py       WoE encoding, fit on training rows only
src/scorecard.py     coefficients to points, adverse action reasons
src/evaluate.py      AUC, KS, Gini, calibration
src/baseline.py      LightGBM comparison
src/policy.py        cutoff economics and sensitivity
src/train.py         entry point
tests/               84 tests on synthetic frames, no dataset required
data/, artifacts/    gitignored
```

## Known limits

**No out-of-time validation.** PSI between train and test is under 0.001, but that
reflects a random split of one population, not two time periods. The dataset has no
date column, so whether the model survives change over time cannot be tested here,
and that is the main risk for a deployed credit model.

**`NumberOfTimes90DaysLate` has no stated time window**, while the 30-59 and 60-89
counters both say "in the last 2 years". Read literally it is a lifetime count. The
observed bad rates, 33.7% at one event rising to 61.7% at three or more rather than
near 100%, say it does not overlap the target period, but that rests on inference
rather than documentation. It is the second strongest feature, so for production
this is a question for whoever owns the data.

**`age` is in the model** at IV 0.26, and it is a protected attribute in many
jurisdictions. Whether to use it is a policy decision rather than a modelling one.

## Status

- [x] EDA and data quality analysis
- [x] Preprocessing extracted from the notebook, with tests
- [x] WoE binning, logistic regression, PDO scorecard
- [x] Gradient boosting baseline, to price what staying interpretable costs
- [x] Approval cutoff analysis against a loss assumption
- [x] CI running the test suite
- [ ] FastAPI scoring service, Streamlit reviewer UI, Docker Compose
