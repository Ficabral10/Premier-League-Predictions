"""Scoreline-level evaluation of the goals models (Poisson vs Dixon-Coles).

Outcome metrics (accuracy, RPS) barely move with Dixon-Coles because it only reshuffles probability
among the four low scores. This script asks the question the correction is meant to answer: how much
probability does each model put on the exact score that happened, and are 0-0, 1-0, 0-1 and 1-1
predicted as often as they occur?

Run:  python -m plpred.scorelines
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import FIRST_TEST_YEAR, RESULTS_DIR, TRAIN_START_YEAR
from .features import build_features, feature_columns
from .models import MAX_GOALS, DixonColesScoreModel, PoissonScoreModel

LOW_SCORES = [(0, 0), (1, 0), (0, 1), (1, 1)]


def evaluate_scorelines(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = build_features() if df is None else df
    df = df[df.season_year >= TRAIN_START_YEAR].reset_index(drop=True)
    features = feature_columns(df)
    rows, cells = [], []
    for year in sorted(y for y in df.season_year.unique() if y >= FIRST_TEST_YEAR):
        train, test = df[df.season_year < year], df[df.season_year == year]
        h = np.clip(test.FTHG.to_numpy(), 0, MAX_GOALS)
        a = np.clip(test.FTAG.to_numpy(), 0, MAX_GOALS)
        for model in (PoissonScoreModel(), DixonColesScoreModel()):
            model.fit(train, features)
            lh, la = model.expected_goals(test)
            mats = np.stack([model.score_matrix(x, y) for x, y in zip(lh, la)])
            p_actual = mats[np.arange(len(test)), h, a]
            rows.append({"model": model.name, "season": test.season.iloc[0],
                         "rho": getattr(model, "rho_", 0.0),
                         "score_log_loss": float(-np.log(p_actual).mean())})
            for i, j in LOW_SCORES:
                cells.append({"model": model.name, "score": f"{i}-{j}",
                              "predicted": float(mats[:, i, j].sum()),
                              "actual": int(((h == i) & (a == j)).sum())})
    return pd.DataFrame(rows), pd.DataFrame(cells).groupby(["model", "score"], sort=False).sum().reset_index()


def main() -> None:
    by_season, cells = evaluate_scorelines()
    RESULTS_DIR.mkdir(exist_ok=True)
    by_season.to_csv(RESULTS_DIR / "scoreline_by_season.csv", index=False)
    cells.to_csv(RESULTS_DIR / "low_score_calibration.csv", index=False)
    print(by_season.groupby("model", sort=False)[["score_log_loss", "rho"]].mean().round(4).to_string())
    print()
    print(cells.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
