"""Interactive demo:  streamlit run app/streamlit_app.py"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from plpred.predict import Predictor

st.set_page_config(page_title="Premier League Predictor", page_icon="⚽", layout="centered")


@st.cache_resource(show_spinner="Training models on 25 seasons of matches...")
def get_predictor() -> Predictor:
    return Predictor()


predictor = get_predictor()
teams = predictor.current_teams

st.title("⚽ Premier League Match Predictor")
st.caption(
    "Ensemble of logistic regression, XGBoost and a Poisson goals model, trained on every "
    "Premier League match since 2000-01. Ratings reflect the latest season in the data "
    f"({predictor.matches.season.max()})."
)

tab_predict, tab_results = st.tabs(["Predict a match", "How accurate is it?"])

with tab_predict:
    c1, c2 = st.columns(2)
    home = c1.selectbox("Home team", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    away_options = [t for t in teams if t != home]
    away = c2.selectbox("Away team", away_options,
                        index=away_options.index("Chelsea") if "Chelsea" in away_options else 0)

    pred = predictor.predict(home, away)

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{home} win", f"{pred.p_home:.0%}")
    m2.metric("Draw", f"{pred.p_draw:.0%}")
    m3.metric(f"{away} win", f"{pred.p_away:.0%}")

    st.markdown(
        f"**Expected goals:** {pred.xg_home:.2f} – {pred.xg_away:.2f} &nbsp;·&nbsp; "
        f"**Most likely score:** {pred.most_likely_score[0]}–{pred.most_likely_score[1]} "
        f"({pred.score_matrix.max():.0%})"
    )
    st.markdown(f"**Elo ratings:** {home} {pred.home_elo:.0f} · {away} {pred.away_elo:.0f}")

    st.subheader("Most likely scorelines")
    m = pred.score_matrix
    flat = [(f"{i}–{j}", m[i, j]) for i in range(6) for j in range(6)]
    top = pd.DataFrame(sorted(flat, key=lambda x: -x[1])[:8], columns=["Score", "Probability"])
    top["Probability"] = top["Probability"].map(lambda v: f"{v:.1%}")
    st.dataframe(top, hide_index=True, use_container_width=True)

    with st.expander("Full scoreline grid (home goals down, away goals across)"):
        grid = pd.DataFrame(np.round(m[:6, :6] * 100, 1), index=range(6), columns=range(6))
        st.dataframe(grid.style.format("{:.1f}%").background_gradient(cmap="Blues"),
                     use_container_width=True)

with tab_results:
    summary_path = ROOT / "results" / "metrics_summary.csv"
    if summary_path.exists():
        st.markdown(
            "Walk-forward backtest: for each season from 2016-17 to 2025-26 the models are "
            "trained only on earlier seasons, then scored on all 380 games."
        )
        st.image(str(ROOT / "figures" / "model_comparison.png"))
        st.image(str(ROOT / "figures" / "accuracy_by_season.png"))
        st.image(str(ROOT / "figures" / "calibration.png"))
    else:
        st.info("Run `python -m plpred.backtest` and `python -m plpred.report` to generate results.")
