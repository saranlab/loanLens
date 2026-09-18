"""Where to set the approval cutoff, and what it costs to be wrong.

    python -m src.policy

AUC says how well the model ranks. It says nothing about where to draw the line,
because that depends on two numbers the model cannot know: what a good loan earns
and what a bad one loses.

Approve an applicant when the expected margin beats the expected loss:

    (1 - p) * margin  >  p * loss

which rearranges to a single threshold on the default probability:

    p  <  margin / (margin + loss)

That is the break-even PD. It depends only on the ratio of loss to margin, not on
the loan size, and it converts to a score cutoff through the same PDO formula the
scorecard uses. Everything here follows from it.

The numbers below are assumptions, not findings. They are stated in one place and
swept, because the honest answer to "where should the cutoff be" is "it moves with
your loss ratio, here is the curve".
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg
from .scorecard import probability_to_score
from .train import DEFAULT_TRAIN_CSV, ROOT, build_dataset, fit_scorecard, split

# Per-loan economics, as a worked assumption. Currency-neutral units.
PRINCIPAL = 10_000.0
MARGIN_RATE = 0.08   # earned over the life of a loan that performs
LGD_RATE = 0.60      # lost on a loan that defaults, after recovery

MARGIN = PRINCIPAL * MARGIN_RATE
LOSS = PRINCIPAL * LGD_RATE


def break_even_pd(margin: float = MARGIN, loss: float = LOSS) -> float:
    """The default probability at which approving breaks even.

    Depends only on the loss-to-margin ratio. At 6,000 against 800 it is 11.8%,
    so an applicant can be quite risky and still be worth approving.
    """
    return margin / (margin + loss)


def break_even_score(margin: float = MARGIN, loss: float = LOSS, **kw) -> float:
    """The break-even PD expressed as a score cutoff."""
    return float(probability_to_score(break_even_pd(margin, loss), **kw))


def profit_at_cutoff(
    scores: pd.Series,
    y: pd.Series,
    cutoff: float,
    margin: float = MARGIN,
    loss: float = LOSS,
) -> dict[str, float]:
    """Book everyone at or above the cutoff and total the result."""
    approved = scores >= cutoff
    n_approved = int(approved.sum())
    bads = int(y[approved].sum())
    goods = n_approved - bads
    return {
        "cutoff": float(cutoff),
        "approval_rate": n_approved / len(scores),
        "n_approved": n_approved,
        "approved_bad_rate": (bads / n_approved) if n_approved else 0.0,
        "profit": goods * margin - bads * loss,
        "profit_per_application": (goods * margin - bads * loss) / len(scores),
    }


def profit_curve(
    scores: pd.Series,
    y: pd.Series,
    margin: float = MARGIN,
    loss: float = LOSS,
    step: float = 2.0,
) -> pd.DataFrame:
    """Profit across the whole range of possible cutoffs."""
    lo = np.floor(scores.min() / step) * step
    hi = np.ceil(scores.max() / step) * step
    grid = np.arange(lo, hi + step, step)
    return pd.DataFrame(
        [profit_at_cutoff(scores, y, c, margin, loss) for c in grid]
    )


def sensitivity(
    scores: pd.Series,
    y: pd.Series,
    loss_ratios=(2.5, 5.0, 7.5, 10.0, 15.0, 25.0),
    margin: float = MARGIN,
) -> pd.DataFrame:
    """How the best cutoff moves as the loss-to-margin ratio changes.

    The point of the table: the cutoff is a business input. A lender with thin
    margins or poor recovery has to cut higher, and no amount of model work
    changes that.
    """
    rows = []
    for ratio in loss_ratios:
        loss = margin * ratio
        curve = profit_curve(scores, y, margin, loss)
        best = curve.loc[curve["profit"].idxmax()]
        rows.append(
            {
                "loss_to_margin": ratio,
                "break_even_pd": break_even_pd(margin, loss),
                "theoretical_cutoff": break_even_score(margin, loss),
                "empirical_cutoff": best["cutoff"],
                "approval_rate": best["approval_rate"],
                "approved_bad_rate": best["approved_bad_rate"],
                "profit_per_application": best["profit_per_application"],
            }
        )
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_TRAIN_CSV)
    args = parser.parse_args(argv)

    if not args.csv.exists():
        parser.error(f"{args.csv} not found. See README for the Kaggle download.")

    X, y = build_dataset(args.csv)
    X_tr, X_te, y_tr, y_te = split(X, y)
    card, _ = fit_scorecard(X_tr, y_tr)
    scores = card.score(X_te)

    print(f"assumptions: principal {PRINCIPAL:,.0f}, margin {MARGIN:,.0f} "
          f"({MARGIN_RATE:.0%}), loss {LOSS:,.0f} ({LGD_RATE:.0%})")
    print(f"loss-to-margin ratio {LOSS / MARGIN:.1f}")
    print(f"break-even PD {break_even_pd():.4f}  ->  score {break_even_score():.1f}")

    curve = profit_curve(scores, y_te)
    best = curve.loc[curve["profit"].idxmax()]

    print(f"\nbest cutoff found by sweep: {best['cutoff']:.0f}")
    print(f"  approval rate      {best['approval_rate']:.2%}")
    print(f"  bad rate approved  {best['approved_bad_rate']:.2%}")
    print(f"  profit per application {best['profit_per_application']:,.1f}")

    # The formula assumes the predicted PD is calibrated. The sweep assumes
    # nothing, so the gap between them is a calibration read rather than an error
    # in either. A few points is expected given the calibration table in train.py,
    # where the model runs slightly cautious at the safe end. A large gap would
    # say the PD is trustworthy as a ranking but not as a level.
    drift = best["cutoff"] - break_even_score()
    print(f"\nformula {break_even_score():.1f}, sweep {best['cutoff']:.0f}, "
          f"gap {drift:+.1f} points")
    print("  the gap is a calibration read: the formula trusts the PD level, "
          "the sweep does not")

    print("\nprofit against cutoff, every 20 points")
    shown = curve[curve["cutoff"] % 20 == 0]
    print(
        shown[["cutoff", "approval_rate", "approved_bad_rate",
               "profit_per_application"]].round(4).to_string(index=False)
    )

    print("\nwhat happens at the tier cutoffs")
    for pd_max, name in cfg.TIER_MAX_PD:
        cut = float(probability_to_score(pd_max))
        at = profit_at_cutoff(scores, y_te, cut)
        print(f"  tier {name} floor ({cut:.0f}): approve {at['approval_rate']:.2%}, "
              f"bad {at['approved_bad_rate']:.2%}, "
              f"profit/app {at['profit_per_application']:,.1f}")

    tier_b_floor = float(probability_to_score(cfg.TIER_MAX_PD[-1][0]))
    forgone = (
        best["profit_per_application"]
        - profit_at_cutoff(scores, y_te, tier_b_floor)["profit_per_application"]
    )
    print(
        f"\nusing the tier B floor ({tier_b_floor:.0f}) as the approval line rather "
        f"than the profit optimum ({best['cutoff']:.0f}) costs {forgone:,.1f} per "
        f"application, {forgone / best['profit_per_application']:.1%}"
    )
    print("  a tier states a risk level, a cutoff trades margin against loss. "
          "Different questions, so conflating them has a price")

    print("\nsensitivity to the loss-to-margin ratio")
    print(sensitivity(scores, y_te).round(4).to_string(index=False))

    out = ROOT / "artifacts" / "profit_curve.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
