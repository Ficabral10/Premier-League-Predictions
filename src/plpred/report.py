"""Make the README charts from backtest results.

Run after the backtest:  python -m plpred.report
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .backtest import RESULTS_DIR, TRAIN_START_YEAR
from .features import build_features, feature_columns
from .models import build_models

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"

# Validated reference palette (light mode); identity never relies on color alone.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

NICE = {
    "ensemble": "Ensemble", "poisson": "Poisson goals", "logreg": "Logistic regression",
    "random_forest": "Random forest", "xgboost": "XGBoost", "elo": "Elo only",
    "baseline": "Baseline (frequencies)",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
})


def _bar_panel(ax, summary, col, title, fmt, lower_is_better):
    s = summary.sort_values("rps")  # same row order in every panel
    best = s[col].min() if lower_is_better else s[col].max()
    colors = [BLUE if v == best else MUTED for v in s[col]]
    y = np.arange(len(s))
    ax.barh(y, s[col], color=colors, height=0.6)
    ax.set_yticks(y, [NICE[m] for m in s.model])
    ax.invert_yaxis()
    lo = s[col].min() * 0.97 if lower_is_better else s[col].min() * 0.95
    ax.set_xlim(lo, s[col].max() * 1.02)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y, s[col]):
        ax.text(v, yi, f" {v:{fmt}}", va="center", color=INK, fontsize=9)
    ax.set_title(title, loc="left", fontsize=11, color=INK)


def model_comparison(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    _bar_panel(axes[0], summary, "accuracy", "Accuracy (higher is better)", ".1%", False)
    _bar_panel(axes[1], summary, "rps", "Ranked probability score (lower is better)", ".4f", True)
    fig.suptitle("3,800 test matches, 2016-17 to 2025-26 (walk-forward)", x=0.01, ha="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)


def accuracy_by_season(by_season: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    series = [("ensemble", BLUE, "o"), ("elo", ORANGE, "s"), ("baseline", AQUA, "^")]
    for key, color, marker in series:
        d = by_season[by_season.model == key]
        ax.plot(d.season, d.accuracy, color=color, lw=2, marker=marker, ms=6, label=NICE[key])
        nudge = {"ensemble": 6, "elo": -6}.get(key, 0)
        ax.annotate(NICE[key], (d.season.iloc[-1], d.accuracy.iloc[-1]), xytext=(8, nudge),
                    textcoords="offset points", va="center", color=INK_2, fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_ylabel("Accuracy")
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    fig.suptitle("Accuracy by test season", x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.set_xlim(-0.3, len(by_season.season.unique()) + 1.2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "accuracy_by_season.png", dpi=160)
    plt.close(fig)


def calibration(preds: pd.DataFrame) -> None:
    p = preds[preds.model == "ensemble"]
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="Perfect calibration")
    bins = np.linspace(0, 1, 11)
    for col, label, cls, color, marker in [("p_home", "Home win", 0, BLUE, "o"),
                                           ("p_draw", "Draw", 1, ORANGE, "s"),
                                           ("p_away", "Away win", 2, AQUA, "^")]:
        idx = np.digitize(p[col], bins) - 1
        g = pd.DataFrame({"pred": p[col], "hit": (p.y == cls).astype(int), "b": idx}).groupby("b")
        agg = g.agg(pred=("pred", "mean"), hit=("hit", "mean"), n=("hit", "size"))
        agg = agg[agg.n >= 30]
        ax.plot(agg.pred, agg.hit, color=color, lw=2, marker=marker, ms=6, label=label)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_xlim(0, 0.9)
    ax.set_ylim(0, 0.9)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Calibration (ensemble)", loc="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration.png", dpi=160)
    plt.close(fig)


def feature_importance(top: int = 12) -> None:
    df = build_features()
    df = df[df.season_year >= TRAIN_START_YEAR]
    feats = feature_columns(df)
    model = build_models()["xgboost"].fit(df, feats)
    booster = model.estimator.get_booster()
    gain = pd.Series(booster.get_score(importance_type="gain")).sort_values(ascending=False).head(top)
    gain = gain / gain.sum()
    fig, ax = plt.subplots(figsize=(7, 4.4))
    y = np.arange(len(gain))
    ax.barh(y, gain.values, color=BLUE, height=0.6)
    ax.set_yticks(y, gain.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of total gain among top features")
    ax.set_title("What the XGBoost model relies on", loc="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    summary = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")
    by_season = pd.read_csv(RESULTS_DIR / "metrics_by_season.csv")
    preds = pd.read_csv(RESULTS_DIR / "predictions.csv.gz")
    model_comparison(summary)
    accuracy_by_season(by_season)
    calibration(preds)
    feature_importance()
    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    main()
