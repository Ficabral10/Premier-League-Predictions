"""Grid-search Elo hyperparameters using ONLY seasons before the backtest window.

Fit on 2000-01..2010-11, score on 2011-12..2015-16 (log loss of an Elo-only model),
so the 2016-17+ test seasons are never touched during tuning.

Run:  python -m plpred.tune_elo
"""

from __future__ import annotations

import itertools

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from .data import load_matches
from .elo import EloConfig, add_elo_features
from .features import OUTCOME_TO_LABEL

GRID = {
    "k": [8, 10, 12, 15, 20, 25],
    "home_advantage": [40, 60],
    "season_regression": [0.0, 0.05, 0.1, 0.2, 0.33],
}


def score_config(matches: pd.DataFrame, config: EloConfig) -> float:
    df, _ = add_elo_features(matches, config)
    df["y"] = df["FTR"].map(OUTCOME_TO_LABEL)
    fit = df[(df.season_year >= 2000) & (df.season_year <= 2010)]
    val = df[(df.season_year >= 2011) & (df.season_year <= 2015)]
    model = LogisticRegression(max_iter=1000).fit(fit[["elo_diff"]], fit["y"])
    return log_loss(val["y"], model.predict_proba(val[["elo_diff"]]), labels=[0, 1, 2])


def main() -> None:
    matches = load_matches()
    rows = []
    for k, ha, reg in itertools.product(*GRID.values()):
        cfg = EloConfig(k=k, home_advantage=ha, season_regression=reg)
        rows.append({"k": k, "home_advantage": ha, "season_regression": reg,
                     "val_log_loss": score_config(matches, cfg)})
    res = pd.DataFrame(rows).sort_values("val_log_loss")
    print(res.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
