"""Load and clean raw Premier League match data (football-data.co.uk via datasets/football-datasets)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Columns we rely on. Shot/corner stats only exist from 2000/01 onward.
CORE_COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
STAT_COLS = ["HS", "AS", "HST", "AST", "HC", "AC"]


def season_label(filename: str) -> str:
    """'season-0506.csv' -> '2005-06'; 'season-9394.csv' -> '1993-94'."""
    code = Path(filename).stem.split("-")[1]
    start = int(code[:2])
    start_year = 1900 + start if start >= 90 else 2000 + start
    return f"{start_year}-{code[2:]}"


def season_start_year(season: str) -> int:
    return int(season[:4])


def load_matches(raw_dir: Path | str = RAW_DIR) -> pd.DataFrame:
    """Return every match, one row per game, sorted chronologically.

    Adds `season` ("2005-06") and `season_year` (2005). Missing stat columns
    in early seasons are left as NaN rather than invented.
    """
    raw_dir = Path(raw_dir)
    frames = []
    for path in sorted(raw_dir.glob("season-*.csv")):
        df = pd.read_csv(path)
        for col in STAT_COLS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[CORE_COLS + STAT_COLS].copy()
        df["season"] = season_label(path.name)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No season-*.csv files found in {raw_dir}")

    matches = pd.concat(frames, ignore_index=True)
    matches["Date"] = pd.to_datetime(matches["Date"])
    matches["season_year"] = matches["season"].map(season_start_year)
    for col in ["FTHG", "FTAG"] + STAT_COLS:
        matches[col] = pd.to_numeric(matches[col], errors="coerce")
    matches = matches.dropna(subset=["FTHG", "FTAG", "FTR"])
    matches[["FTHG", "FTAG"]] = matches[["FTHG", "FTAG"]].astype(int)
    # Stable chronological order; ties broken by original file order.
    matches = matches.sort_values(["Date"], kind="stable").reset_index(drop=True)
    matches["match_id"] = matches.index
    return matches
