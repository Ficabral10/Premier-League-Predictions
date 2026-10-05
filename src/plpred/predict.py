"""Predict upcoming fixtures from the latest available data.

    python -m plpred.predict "Arsenal" "Chelsea"

How it works: the fixture is appended as a placeholder row dated after the last
played match. Because every feature only looks at *earlier* games, the placeholder's
features are exactly each team's current Elo rating and recent form; its own
(fake) result is never used.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import TRAIN_START_YEAR
from .data import load_matches
from .features import build_features, feature_columns
from .models import AverageEnsemble, PoissonScoreModel, build_models


@dataclass
class FixturePrediction:
    home: str
    away: str
    p_home: float
    p_draw: float
    p_away: float
    xg_home: float
    xg_away: float
    most_likely_score: tuple[int, int]
    score_matrix: np.ndarray
    home_elo: float
    away_elo: float


class Predictor:
    """Trains the ensemble on all history once, then predicts any fixture."""

    def __init__(self):
        self.matches = load_matches()
        self.train = build_features(self.matches)
        self.train = self.train[self.train.season_year >= TRAIN_START_YEAR]
        self.features = feature_columns(self.train)
        models = build_models()
        self.ensemble: AverageEnsemble = models["ensemble"].fit(self.train, self.features)
        self.poisson: PoissonScoreModel = next(
            m for m in self.ensemble.members if isinstance(m, PoissonScoreModel)
        )

    @property
    def current_teams(self) -> list[str]:
        last = self.matches[self.matches.season == self.matches.season.max()]
        return sorted(set(last.HomeTeam) | set(last.AwayTeam))

    def _fixture_rows(self, fixtures: list[tuple[str, str]]) -> pd.DataFrame:
        last = self.matches.iloc[-1]
        rows = pd.DataFrame({
            "Date": last.Date + pd.Timedelta(days=7), "HomeTeam": [h for h, _ in fixtures],
            "AwayTeam": [a for _, a in fixtures], "FTHG": 0, "FTAG": 0, "FTR": "D",
            "season": last.season, "season_year": last.season_year,
        })
        combined = pd.concat([self.matches, rows], ignore_index=True)
        combined["match_id"] = combined.index
        feats = build_features(combined)
        return feats[feats.match_id >= len(self.matches)].sort_values("match_id")

    def predict(self, home: str, away: str) -> FixturePrediction:
        if home == away:
            raise ValueError("Pick two different teams.")
        row = self._fixture_rows([(home, away)])
        p = self.ensemble.predict_proba(row)[0]
        lh, la = self.poisson.expected_goals(row)
        matrix = self.poisson.score_matrix(lh[0], la[0])
        i, j = np.unravel_index(matrix.argmax(), matrix.shape)
        return FixturePrediction(
            home, away, float(p[0]), float(p[1]), float(p[2]), float(lh[0]), float(la[0]),
            (int(i), int(j)), matrix, float(row.home_elo.iloc[0]), float(row.away_elo.iloc[0]),
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Predict a Premier League fixture")
    parser.add_argument("home")
    parser.add_argument("away")
    args = parser.parse_args(argv)
    pred = Predictor().predict(args.home, args.away)
    print(f"{pred.home} vs {pred.away}")
    print(f"  Home win {pred.p_home:.1%} | Draw {pred.p_draw:.1%} | Away win {pred.p_away:.1%}")
    print(f"  Expected goals {pred.xg_home:.2f} - {pred.xg_away:.2f}; "
          f"most likely score {pred.most_likely_score[0]}-{pred.most_likely_score[1]}")


if __name__ == "__main__":
    main()
