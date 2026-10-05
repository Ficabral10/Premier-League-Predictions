"""Elo ratings for football, updated match by match.

Design choices (each is a tunable parameter):
- Home advantage is added to the home side's rating when computing the expected score.
- Updates scale with the goal margin (a 4-0 moves ratings more than a 1-0).
- Between seasons, ratings regress part of the way back to the mean.
- Promoted teams with no history start below average.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class EloConfig:
    k: float = 15.0          # tuned on 2000-2015 only (see tune_elo.py)
    home_advantage: float = 40.0
    initial_rating: float = 1500.0
    promoted_rating: float = 1420.0
    season_regression: float = 0.1  # fraction pulled back toward the mean each new season


@dataclass
class EloTracker:
    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)
    current_season: str | None = None
    first_season: str | None = None

    def rating(self, team: str) -> float:
        if team not in self.ratings:
            # Teams in the very first season start at the mean; any later
            # newcomer has been promoted from the Championship.
            in_first_season = self.current_season is None or self.current_season == self.first_season
            self.ratings[team] = (
                self.config.initial_rating if in_first_season else self.config.promoted_rating
            )
        return self.ratings[team]

    def start_season(self, season: str) -> None:
        if self.first_season is None:
            self.first_season = season
        if self.current_season is not None and season != self.current_season and self.ratings:
            mean = float(np.mean(list(self.ratings.values())))
            r = self.config.season_regression
            self.ratings = {t: (1 - r) * v + r * mean for t, v in self.ratings.items()}
        self.current_season = season

    def expected_home(self, home: str, away: str) -> float:
        diff = self.rating(home) + self.config.home_advantage - self.rating(away)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        exp_home = self.expected_home(home, away)
        actual = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        margin = abs(home_goals - away_goals)
        multiplier = 1.0 if margin <= 1 else (1.5 if margin == 2 else (11 + margin) / 8)
        delta = self.config.k * multiplier * (actual - exp_home)
        self.ratings[home] += delta
        self.ratings[away] -= delta


def add_elo_features(matches: pd.DataFrame, config: EloConfig | None = None) -> tuple[pd.DataFrame, EloTracker]:
    """Attach pre-match Elo ratings to every match.

    The ratings stored on a row are the values *before* that match is played,
    so they never contain information from the match itself.
    """
    tracker = EloTracker(config or EloConfig())
    home_elo, away_elo, exp_home = [], [], []
    for row in matches.itertuples(index=False):
        tracker.start_season(row.season)
        home_elo.append(tracker.rating(row.HomeTeam))
        away_elo.append(tracker.rating(row.AwayTeam))
        exp_home.append(tracker.expected_home(row.HomeTeam, row.AwayTeam))
        tracker.update(row.HomeTeam, row.AwayTeam, row.FTHG, row.FTAG)
    out = matches.copy()
    out["home_elo"] = home_elo
    out["away_elo"] = away_elo
    out["elo_diff"] = out["home_elo"] - out["away_elo"]
    out["elo_exp_home"] = exp_home
    return out, tracker
