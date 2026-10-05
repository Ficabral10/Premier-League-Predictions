"""Match-outcome models. Every model exposes fit(train_df) and predict_proba(df) -> (n, 3)
with columns in [Home win, Draw, Away win] order."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

MAX_GOALS = 10


class PriorBaseline:
    """Ignores the teams entirely: predicts historical H/D/A frequencies every game."""

    name = "Baseline (historical frequencies)"

    def fit(self, train: pd.DataFrame, features: list[str]) -> PriorBaseline:
        counts = np.bincount(train["y"], minlength=3)
        self.proba_ = counts / counts.sum()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.tile(self.proba_, (len(df), 1))


class SklearnOutcomeModel:
    def __init__(self, name: str, estimator, feature_subset: list[str] | None = None):
        self.name = name
        self.estimator = estimator
        self.feature_subset = feature_subset

    def fit(self, train: pd.DataFrame, features: list[str]) -> SklearnOutcomeModel:
        self.features_ = self.feature_subset or features
        self.estimator.fit(train[self.features_], train["y"])
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict_proba(df[self.features_])


class PoissonScoreModel:
    """Predicts expected goals for each side with two Poisson regressions, then turns them
    into a full scoreline probability matrix (assumes the two goal counts are independent).
    Gives outcome probabilities *and* the most likely exact score."""

    name = "Poisson goals model"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _pipe(self):
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(), PoissonRegressor(alpha=self.alpha, max_iter=1000)
        )

    def fit(self, train: pd.DataFrame, features: list[str]) -> PoissonScoreModel:
        self.features_ = features
        self.home_ = self._pipe().fit(train[features], train["FTHG"])
        self.away_ = self._pipe().fit(train[features], train["FTAG"])
        return self

    def expected_goals(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.home_.predict(df[self.features_]), self.away_.predict(df[self.features_])

    @staticmethod
    def score_matrix(lam_home: float, lam_away: float) -> np.ndarray:
        goals = np.arange(MAX_GOALS + 1)
        m = np.outer(poisson.pmf(goals, lam_home), poisson.pmf(goals, lam_away))
        return m / m.sum()

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        lh, la = self.expected_goals(df)
        out = np.empty((len(df), 3))
        for i, (h, a) in enumerate(zip(lh, la)):
            m = self.score_matrix(h, a)
            out[i] = [np.tril(m, -1).sum(), np.trace(m), np.triu(m, 1).sum()]
        return out

    def most_likely_score(self, df: pd.DataFrame) -> list[tuple[int, int]]:
        lh, la = self.expected_goals(df)
        scores = []
        for h, a in zip(lh, la):
            m = self.score_matrix(h, a)
            i, j = np.unravel_index(m.argmax(), m.shape)
            scores.append((int(i), int(j)))
        return scores


class AverageEnsemble:
    """Averages the probabilities of several fitted models."""

    def __init__(self, name: str, members: list):
        self.name = name
        self.members = members

    def fit(self, train: pd.DataFrame, features: list[str]) -> AverageEnsemble:
        for m in self.members:
            m.fit(train, features)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return np.mean([m.predict_proba(df) for m in self.members], axis=0)


def _logreg(C: float = 0.1):
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=C, max_iter=2000)
    )


def build_models(random_state: int = 42) -> dict[str, object]:
    """Fresh, unfitted instances of every model in the comparison."""
    xgb = lambda: XGBClassifier(
        n_estimators=400, max_depth=3, learning_rate=0.02, subsample=0.8, colsample_bytree=0.7,
        min_child_weight=10, reg_lambda=5.0, objective="multi:softprob", eval_metric="mlogloss",
        random_state=random_state, n_jobs=4,
    )
    return {
        "baseline": PriorBaseline(),
        "elo": SklearnOutcomeModel("Elo rating only", _logreg(C=1.0), feature_subset=["elo_diff"]),
        "logreg": SklearnOutcomeModel("Logistic regression", _logreg()),
        "random_forest": SklearnOutcomeModel(
            "Random forest",
            make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=500, min_samples_leaf=25, max_features="sqrt",
                    random_state=random_state, n_jobs=4,
                ),
            ),
        ),
        "xgboost": SklearnOutcomeModel("XGBoost", xgb()),
        "poisson": PoissonScoreModel(),
        "ensemble": AverageEnsemble(
            "Ensemble (LogReg + XGBoost + Poisson)",
            [SklearnOutcomeModel("LogReg", _logreg()), SklearnOutcomeModel("XGB", xgb()), PoissonScoreModel()],
        ),
    }
