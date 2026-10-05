"""Walk-forward backtest: for each test season, train only on earlier seasons.

This mirrors real use (you only ever know the past) and avoids the classic mistake of
randomly shuffling matches, which leaks future information into training.

Run:  python -m plpred.backtest
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import build_features, feature_columns
from .metrics import evaluate
from .models import PoissonScoreModel, build_models

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

TRAIN_START_YEAR = 2000  # shot statistics begin in 2000-01
FIRST_TEST_YEAR = 2016   # test on the last 10 seasons: 2016-17 ... 2025-26


def walk_forward(df: pd.DataFrame, first_test_year: int = FIRST_TEST_YEAR, models: list[str] | None = None):
    df = df[df.season_year >= TRAIN_START_YEAR].reset_index(drop=True)
    features = feature_columns(df)
    test_years = sorted(y for y in df.season_year.unique() if y >= first_test_year)

    season_rows, pred_frames = [], []
    for year in test_years:
        train, test = df[df.season_year < year], df[df.season_year == year]
        fitted = build_models()
        for key, model in fitted.items():
            if models and key not in models:
                continue
            model.fit(train, features)
            proba = model.predict_proba(test)
            season_rows.append({"model": key, "name": model.name, "season": test.season.iloc[0],
                                "n": len(test), **evaluate(test["y"].to_numpy(), proba)})
            frame = test[["match_id", "Date", "season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "y"]].copy()
            frame["model"] = key
            frame[["p_home", "p_draw", "p_away"]] = proba
            if isinstance(model, PoissonScoreModel):
                lh, la = model.expected_goals(test)
                frame["xg_home"], frame["xg_away"] = lh, la
                frame[["pred_hg", "pred_ag"]] = np.array(model.most_likely_score(test))
            pred_frames.append(frame)
        print(f"finished {test.season.iloc[0]} (train n={len(train)}, test n={len(test)})")

    by_season = pd.DataFrame(season_rows)
    preds = pd.concat(pred_frames, ignore_index=True)
    return by_season, preds


def summarize(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key, g in preds.groupby("model", sort=False):
        proba = g[["p_home", "p_draw", "p_away"]].to_numpy()
        rows.append({"model": key, "n": len(g), **evaluate(g["y"].to_numpy(), proba),
                     "draws_predicted": float((proba.argmax(1) == 1).mean())})
    return pd.DataFrame(rows).sort_values("rps").reset_index(drop=True)


def score_metrics(preds: pd.DataFrame) -> dict[str, float]:
    p = preds[preds.model == "poisson"]
    return {
        "exact_score_accuracy": float(((p.pred_hg == p.FTHG) & (p.pred_ag == p.FTAG)).mean()),
        "home_goals_mae": float((p.xg_home - p.FTHG).abs().mean()),
        "away_goals_mae": float((p.xg_away - p.FTAG).abs().mean()),
        "total_goals_mae": float(((p.xg_home + p.xg_away) - (p.FTHG + p.FTAG)).abs().mean()),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-test-year", type=int, default=FIRST_TEST_YEAR)
    args = parser.parse_args(argv)

    df = build_features()
    by_season, preds = walk_forward(df, args.first_test_year)
    summary = summarize(preds)
    scores = score_metrics(preds)

    RESULTS_DIR.mkdir(exist_ok=True)
    by_season.to_csv(RESULTS_DIR / "metrics_by_season.csv", index=False)
    summary.to_csv(RESULTS_DIR / "metrics_summary.csv", index=False)
    preds.to_csv(RESULTS_DIR / "predictions.csv.gz", index=False)
    pd.Series(scores).to_csv(RESULTS_DIR / "score_metrics.csv", header=["value"])

    pd.set_option("display.width", 140)
    print("\nOverall (all test seasons pooled):")
    print(summary.round(4).to_string(index=False))
    print("\nPoisson scoreline metrics:", {k: round(v, 4) for k, v in scores.items()})


if __name__ == "__main__":
    main()
