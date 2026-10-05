import numpy as np
import pandas as pd
import pytest

from plpred.data import load_matches, season_label
from plpred.elo import EloConfig, EloTracker
from plpred.features import build_features, feature_columns
from plpred.metrics import evaluate, ranked_probability_score
from plpred.models import PoissonScoreModel, PriorBaseline


@pytest.fixture(scope="module")
def matches():
    return load_matches()


@pytest.fixture(scope="module")
def features(matches):
    return build_features(matches)


def test_season_label():
    assert season_label("season-9394.csv") == "1993-94"
    assert season_label("season-0506.csv") == "2005-06"
    assert season_label("season-2526.csv") == "2025-26"


def test_matches_loaded_and_sorted(matches):
    assert len(matches) > 12_000
    assert matches["Date"].is_monotonic_increasing
    assert set(matches["FTR"]) == {"H", "D", "A"}
    # Every full season has 380 games (the 22-team early-90s seasons had 462).
    assert set(matches.groupby("season").size()) <= {380, 462}


def test_elo_is_zero_sum_within_a_season():
    tracker = EloTracker(EloConfig())
    tracker.start_season("2000-01")
    for t in ["A", "B", "C", "D"]:
        tracker.rating(t)
    before = sum(tracker.ratings.values())
    tracker.update("A", "B", 3, 0)
    tracker.update("C", "D", 1, 1)
    assert sum(tracker.ratings.values()) == pytest.approx(before)


def test_elo_winner_gains_rating():
    tracker = EloTracker(EloConfig())
    tracker.start_season("2000-01")
    tracker.update("A", "B", 2, 0)
    assert tracker.ratings["A"] > tracker.ratings["B"]


def test_promoted_teams_start_below_average():
    tracker = EloTracker(EloConfig())
    tracker.start_season("2000-01")
    assert tracker.rating("Founding") == 1500
    tracker.start_season("2001-02")
    assert tracker.rating("Promoted") == EloConfig().promoted_rating


def test_no_leakage_features_ignore_the_match_itself(matches):
    """Changing a match's result must not change that match's own features."""
    base = build_features(matches)
    target = base[base.season == "2015-16"].iloc[100]
    altered = matches.copy()
    idx = altered.index[altered.match_id == target.match_id][0]
    altered.loc[idx, ["FTHG", "FTAG", "FTR"]] = [9, 0, "H"]
    changed = build_features(altered)
    cols = feature_columns(base)
    a = base.loc[base.match_id == target.match_id, cols].iloc[0]
    b = changed.loc[changed.match_id == target.match_id, cols].iloc[0]
    pd.testing.assert_series_equal(a, b, check_names=False)


def test_feature_rows_match_games(features, matches):
    assert len(features) == len(matches)
    assert features["y"].isin([0, 1, 2]).all()


def test_rps_perfect_and_worst():
    y = np.array([0, 2])
    perfect = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
    worst = np.array([[0, 0, 1], [1, 0, 0]], dtype=float)
    assert ranked_probability_score(y, perfect) == 0
    assert ranked_probability_score(y, worst) == 1


def test_rps_rewards_near_misses():
    y = np.array([0])  # home win
    near = np.array([[0, 1, 0]], dtype=float)  # predicted draw
    far = np.array([[0, 0, 1]], dtype=float)   # predicted away win
    assert ranked_probability_score(y, near) < ranked_probability_score(y, far)


def test_poisson_score_matrix_is_a_distribution():
    m = PoissonScoreModel.score_matrix(1.6, 1.1)
    assert m.sum() == pytest.approx(1.0)
    assert (m >= 0).all()
    # Stronger expected attack means a higher chance of winning
    strong = PoissonScoreModel.score_matrix(2.5, 0.7)
    assert np.tril(strong, -1).sum() > np.tril(m, -1).sum()


def test_models_return_valid_probabilities(features):
    df = features[features.season_year >= 2010]
    train, test = df[df.season_year < 2014], df[df.season_year == 2014]
    cols = feature_columns(df)
    for model in (PriorBaseline(), PoissonScoreModel()):
        proba = model.fit(train, cols).predict_proba(test)
        assert proba.shape == (len(test), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        metrics = evaluate(test["y"].to_numpy(), proba)
        assert 0 <= metrics["accuracy"] <= 1
