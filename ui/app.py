"""Reviewer interface for the scoring service.

    streamlit run ui/app.py

Holds no model and does no arithmetic. Everything shown here comes from the API,
so there is exactly one implementation of the scorecard and the screen cannot
drift from the decision.

Written for a credit officer rather than a data scientist: the score, the
decision, and what cost the applicant the most points. No AUC, no WoE.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
TIMEOUT = 10

st.set_page_config(page_title="LoanLens", page_icon="•", layout="wide")


@st.cache_data(ttl=30)
def health() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        return r.json() if r.ok else None
    except requests.RequestException:
        return None


def score(payload: dict) -> tuple[dict | None, str | None]:
    try:
        r = requests.post(f"{API_URL}/score", json=payload, timeout=TIMEOUT)
    except requests.RequestException as exc:
        return None, f"cannot reach the scoring service at {API_URL}: {exc}"

    if r.status_code == 422:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in d['loc'][1:])}: {d['msg']}"
            for d in r.json()["detail"]
        )
        return None, f"the application was rejected by validation: {problems}"
    if not r.ok:
        return None, f"{r.status_code}: {r.json().get('detail', r.text)}"
    return r.json(), None


st.title("LoanLens")
st.caption("Credit scorecard. Every decision comes with the reasons behind it.")

status = health()
if status is None:
    st.error(f"No scoring service at {API_URL}. Start it, then reload this page.")
    st.stop()
if not status["model_loaded"]:
    st.error("The service is up but no scorecard is loaded. Run `python -m src.train`.")
    st.stop()

with st.sidebar:
    st.subheader("Service")
    st.write(f"Features: {status['features']}")
    st.write(f"Approval cutoff: {status['cutoff']:.0f}")
    st.caption(
        "The cutoff is a business input, not a model output. It is the profit "
        "optimum at an assumed 8% margin and 60% loss given default, and it moves "
        "with that ratio."
    )

st.subheader("Application")
left, middle, right = st.columns(3)

with left:
    st.markdown("**Borrower**")
    age = st.number_input("Age", 18, 110, 45)
    dependents_known = st.checkbox("Dependents known", value=True)
    dependents = st.number_input("Dependents", 0, 25, 2, disabled=not dependents_known)
    income_known = st.checkbox("Income known", value=True)
    income = st.number_input(
        "Monthly income", 0.0, 1_000_000.0, 5400.0, step=100.0, disabled=not income_known
    )

with middle:
    st.markdown("**Exposure**")
    utilization = st.slider(
        "Revolving utilization", 0.0, 2.0, 0.25, 0.01,
        help="Card balances over total limits. Above 1.0 means over the limit, "
             "which the data shows is genuinely riskier, not a data error.",
    )
    debt_ratio = st.slider("Debt ratio", 0.0, 3.0, 0.35, 0.01,
                           help="Debt payments, alimony and living costs over gross income.")
    open_lines = st.number_input("Open credit lines and loans", 0, 100, 8)
    real_estate = st.number_input("Real estate loans", 0, 60, 1)

with right:
    st.markdown("**Delinquency history**")
    late_30 = st.number_input("30-59 days past due", 0, 20, 0)
    late_60 = st.number_input("60-89 days past due", 0, 20, 0)
    late_90 = st.number_input("90+ days past due", 0, 20, 0)

payload = {
    "RevolvingUtilizationOfUnsecuredLines": utilization,
    "age": int(age),
    "NumberOfTime30-59DaysPastDueNotWorse": int(late_30),
    "DebtRatio": debt_ratio,
    "NumberOfOpenCreditLinesAndLoans": int(open_lines),
    "NumberOfTimes90DaysLate": int(late_90),
    "NumberRealEstateLoansOrLines": int(real_estate),
    "NumberOfTime60-89DaysPastDueNotWorse": int(late_60),
}
if income_known:
    payload["MonthlyIncome"] = float(income)
if dependents_known:
    payload["NumberOfDependents"] = int(dependents)

if st.button("Assess", type="primary", use_container_width=True):
    result, error = score(payload)
    if error:
        st.error(error)
        st.stop()

    st.divider()
    a, b, c, d = st.columns(4)
    a.metric("Score", f"{result['score']:.0f}")
    b.metric("Probability of default", f"{result['probability_of_default']:.2%}")
    c.metric("Tier", result["tier"])
    d.metric("Cutoff", f"{result['cutoff']:.0f}",
             delta=f"{result['score'] - result['cutoff']:+.0f}")

    if result["decision"] == "approve":
        st.success("Approve")
    else:
        st.error("Decline")

    reasons, breakdown = st.columns([1, 1])

    with reasons:
        st.subheader("Why")
        if result["reasons"]:
            st.caption(
                "The factors costing the most against the best available on each. "
                "These are the adverse action reasons."
            )
            for i, reason in enumerate(result["reasons"], 1):
                st.write(f"**{i}. {reason['feature']}**")
                st.write(f"{reason['points']:+.1f} points, "
                         f"{reason['points_lost']:.1f} below the best bin")
        else:
            st.write("Nothing counts against this application.")

    with breakdown:
        st.subheader("Points")
        st.caption(f"Base {result['base_points']:.1f}, then each factor adds or subtracts.")
        rows = sorted(result["points"].items(), key=lambda kv: kv[1])
        st.bar_chart({"points": dict(rows)}, horizontal=True)

    with st.expander("Full breakdown"):
        st.caption(
            "Base points plus every line equals the score. The API computes this "
            "from the same table, so the screen cannot disagree with the decision."
        )
        st.table(
            [{"factor": k, "points": round(v, 2)} for k, v in rows]
            + [{"factor": "base", "points": round(result["base_points"], 2)},
               {"factor": "TOTAL", "points": round(result["score"], 2)}]
        )

with st.expander("The scorecard"):
    st.caption(
        "The points table the decision is made from. Published because a decision "
        "nobody can check is not one anyone should be making about someone's credit."
    )
    try:
        card = requests.get(f"{API_URL}/scorecard", timeout=TIMEOUT).json()
        for feature in card["features"]:
            st.write(f"**{feature['name']}** (IV {feature['iv']:.3f})")
            rows = [
                {
                    "bin": (
                        f"{b.get('left', '')} to {b.get('right', '')}"
                        if feature["kind"] == "interval"
                        else str(b.get("value"))
                    ),
                    "points": b["points"],
                }
                for b in feature["bins"]
            ]
            rows.append({"bin": "missing", "points": feature["missing_points"]})
            st.table(rows)
    except requests.RequestException as exc:
        st.warning(f"could not load the scorecard: {exc}")
