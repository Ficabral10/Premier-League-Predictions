"""Pre-match features.

Every feature for a match is computed only from matches played *before* it
(rolling windows are shifted by one game), so the model never sees the result
it is trying to predict.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import load_matches
from .elo import EloConfig, add_elo_features

# A gap this long means the team left the league (relegated) and came back;
# form from a previous Premier League stint is discarded.
STINT_GAP_DAYS = 200

ROLLING_STATS = ["pts", "gf", "ga", "shots_for", "shots_against", "sot_for", "sot_against"]
WINDOWS = (5, 10)
EWM_HALFLIFE = 8  # games

OUTCOME_TO_LABEL = {"H": 0, "D": 1, "A": 2}
LABEL_TO_OUTCOME = {v: k for k, v in OUTCOME_TO_LABEL.items()}


def _team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per (match, team) from that team's point of view."""
    home = pd.DataFrame({
        "match_id": matches["match_id"], "Date": matches["Date"], "season": matches["season"],
        "team": matches["HomeTeam"], "is_home": 1,
        "gf": matches["FTHG"], "ga": matches["FTAG"],
        "shots_for": matches["HS"], "shots_against": matches["AS"],
        "sot_for": matches["HST"], "sot_against": matches["AST"],
    })
    away = pd.DataFrame({
        "match_id": matches["match_id"], "Date": matches["Date"], "season": matches["season"],
        "team": matches["AwayTeam"], "is_home": 0,
        "gf": matches["FTAG"], "ga": matches["FTHG"],
        "shots_for": matches["AS"], "shots_against": matches["HS"],
        "sot_for": matches["AST"], "sot_against": matches["HST"],
    })
    long = pd.concat([home, away], ignore_index=True)
    long["pts"] = np.select([long.gf > long.ga, long.gf == long.ga], [3, 1], 0)
    return long.sort_values(["team", "Date", "match_id"]).reset_index(drop=True)


def _rolling_team_features(long: pd.DataFrame) -> pd.DataFrame:
    long = long.copy()
    gap = long.groupby("team")["Date"].diff().dt.days
    long["stint"] = (gap.isna() | (gap > STINT_GAP_DAYS)).groupby(long["team"]).cumsum()
    long["rest_days"] = gap.where(gap <= 30)  # cap: first game after a break is "fully rested"

    grp = long.groupby(["team", "stint"], sort=False)
    feats = long[["match_id", "team", "is_home", "rest_days"]].copy()
    for w in WINDOWS:
        for stat in ROLLING_STATS:
            # shift(1): only games strictly before this one
            feats[f"form{w}_{stat}"] = grp[stat].transform(
                lambda s, w=w: s.shift(1).rolling(w, min_periods=3).mean()
            )
    # Exponentially weighted form: recent games count most, but nothing is thrown away.
    for stat in ("gf", "ga", "sot_for", "sot_against"):
        feats[f"ewm_{stat}"] = grp[stat].transform(
            lambda s: s.shift(1).ewm(halflife=EWM_HALFLIFE, min_periods=3).mean()
        )
    season_grp = long.groupby(["team", "season"], sort=False)
    feats["season_ppg"] = season_grp["pts"].transform(lambda s: s.shift(1).expanding().mean())
    feats["season_games"] = season_grp.cumcount()
    feats["stint_games"] = grp.cumcount()
    return feats


def build_features(matches: pd.DataFrame | None = None, elo_config: EloConfig | None = None) -> pd.DataFrame:
    """Return one row per match with pre-match features for both sides plus targets."""
    if matches is None:
        matches = load_matches()
    matches, _ = add_elo_features(matches, elo_config)

    team_feats = _rolling_team_features(_team_long(matches))
    value_cols = [c for c in team_feats.columns if c not in ("match_id", "team", "is_home")]
    home = team_feats[team_feats.is_home == 1][["match_id"] + value_cols].add_prefix("home_")
    away = team_feats[team_feats.is_home == 0][["match_id"] + value_cols].add_prefix("away_")
    df = (
        matches.merge(home, left_on="match_id", right_on="home_match_id")
        .merge(away, left_on="match_id", right_on="away_match_id")
        .drop(columns=["home_match_id", "away_match_id"])
    )

    for w in WINDOWS:
        df[f"diff_form{w}_pts"] = df[f"home_form{w}_pts"] - df[f"away_form{w}_pts"]
        df[f"diff_form{w}_gd"] = (df[f"home_form{w}_gf"] - df[f"home_form{w}_ga"]) - (
            df[f"away_form{w}_gf"] - df[f"away_form{w}_ga"]
        )
        df[f"diff_form{w}_sot"] = (df[f"home_form{w}_sot_for"] - df[f"home_form{w}_sot_against"]) - (
            df[f"away_form{w}_sot_for"] - df[f"away_form{w}_sot_against"]
        )
    df["diff_ewm_gd"] = (df["home_ewm_gf"] - df["home_ewm_ga"]) - (df["away_ewm_gf"] - df["away_ewm_ga"])
    df["diff_ewm_sot"] = (df["home_ewm_sot_for"] - df["home_ewm_sot_against"]) - (
        df["away_ewm_sot_for"] - df["away_ewm_sot_against"]
    )
    df["diff_season_ppg"] =df["home_season_ppg"] - df["away_season_ppg"]
    df["diff_rest_days"] = df["home_rest_days"] - df["away_rest_days"]

    df["y"] = df["FTR"].map(OUTCOME_TO_LABEL)
    return df.sort_values(["Date", "match_id"]).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model inputs (pre-match information only)."""
    excluded = {
        "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST", "HC", "AC",
        "season", "season_year", "match_id", "y", "elo_exp_home",
    }
    return [c for c in df.columns if c not in excluded]
