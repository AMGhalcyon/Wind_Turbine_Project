"""
Honest model comparison - the source of every accuracy number in the README.

Three tests, from weakest to strongest:
  1. Random 5-fold CV      - flatters every model: the held-out row's load/material twins stay in training
  2. Leave-one-geometry-out - holds out a whole blade shape (all 9 load x material runs of it)
  3. Off-grid ANSYS run    - one brand-new simulation, off-grid in length, chord and load at once

Every model gets the same log-transformed target, so the comparison is like-for-like.

Run:  python compare_models.py      (~2 minutes; writes Visualisations/Notebook_Plots/model_comparison.png)

@author: anish
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib

matplotlib.use("Agg")  # file output only - no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from xgboost import XGBRegressor

from src.data_processing import get_features_and_targets, load_dataset
from src.evaluation import geometry_groups, mape
from src.prediction import predict_new_sample
from src.training import PhysicsInformedGP, PowerLawModel, build_pipeline, make_gpr
from src.utils import OFF_GRID_ANSYS, OFF_GRID_CASE, RANDOM_STATE, REPO_ROOT

FIGURE_PATH = REPO_ROOT / "Visualisations" / "Notebook_Plots" / "model_comparison.png"
FINAL_MODEL = "Physics-informed GP"

# Reference palette (dataviz skill): one highlight hue for the shipped model, neutral for baselines
SURFACE, TEXT, TEXT_SECONDARY, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HIGHLIGHT, NEUTRAL = "#2a78d6", "#b5b4ad"


def log_target(model):
    return build_pipeline(TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp))


MODELS = {  # name -> factory returning an unfitted estimator that accepts the encoded DataFrame
    "Random Forest": lambda: log_target(RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)),
    "XGBoost": lambda: log_target(XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                                               random_state=RANDOM_STATE)),
    "Pure GPR (old final)": lambda: build_pipeline(make_gpr()),
    "Power law (6 params)": PowerLawModel,
    FINAL_MODEL: PhysicsInformedGP,
}


def evaluate(X: pd.DataFrame, y_stress: pd.Series, y_defl: pd.Series) -> pd.DataFrame:
    groups = geometry_groups(X)
    kfold = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, make in MODELS.items():
        print(f"  evaluating {name}...")
        row = {"Model": name}
        for target, y in (("stress", y_stress), ("deflection", y_defl)):
            row[f"{target}_kfold"] = mape(y, cross_val_predict(make(), X, y, cv=kfold))
            row[f"{target}_geometry"] = mape(y, cross_val_predict(make(), X, y, cv=LeaveOneGroupOut(), groups=groups))
        full_stress, full_defl = make().fit(X, y_stress), make().fit(X, y_defl)
        pred = predict_new_sample(full_stress, full_defl, **OFF_GRID_CASE)
        row["stress_offgrid"] = 100 * (pred["stress_MPa"] / OFF_GRID_ANSYS["stress_MPa"] - 1)
        row["deflection_offgrid"] = 100 * (pred["deflection_mm"] / OFF_GRID_ANSYS["deflection_mm"] - 1)
        row["stress_pred"], row["deflection_pred"] = pred["stress_MPa"], pred["deflection_mm"]
        rows.append(row)
    return pd.DataFrame(rows)


def print_markdown(results: pd.DataFrame) -> None:
    print("\n| Model | Random 5-fold (stress / defl.) | Hold out a geometry (stress / defl.) "
          "| Off-grid ANSYS run (stress / defl.) |")
    print("| :--- | :---: | :---: | :---: |")
    for _, r in results.iterrows():
        name = f"**{r.Model}**" if r.Model == FINAL_MODEL else r.Model
        print(f"| {name} | {r.stress_kfold:.2f}% / {r.deflection_kfold:.2f}% "
              f"| {r.stress_geometry:.2f}% / {r.deflection_geometry:.2f}% "
              f"| {r.stress_offgrid:+.1f}% / {r.deflection_offgrid:+.1f}% |")
    final = results.set_index("Model").loc[FINAL_MODEL]
    print(f"\nOff-grid prediction ({FINAL_MODEL}): {final.stress_pred:.2f} MPa, {final.deflection_pred:.2f} mm "
          f"vs ANSYS {OFF_GRID_ANSYS['stress_MPa']} MPa, {OFF_GRID_ANSYS['deflection_mm']} mm")


def plot(results: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, facecolor=SURFACE)
    names = results["Model"].tolist()[::-1]  # final model on top
    colors = [HIGHLIGHT if n == FINAL_MODEL else NEUTRAL for n in names]
    for ax, target, title in ((axes[0], "stress", "Peak stress"), (axes[1], "deflection", "Tip deflection")):
        values = results.set_index("Model").loc[names, f"{target}_geometry"].to_numpy()
        ax.set_facecolor(SURFACE)
        ax.barh(names, values, height=0.55, color=colors)
        ax.set_xscale("log")
        ax.set_xlim(0.05, 400)
        ax.set_xticks([0.1, 1, 10, 100], labels=["0.1%", "1%", "10%", "100%"])
        ax.minorticks_off()
        for y, v in enumerate(values):
            ax.text(v * 1.12, y, f"{v:.2f}%" if v < 10 else f"{v:.0f}%", va="center",
                    fontsize=10, color=TEXT if names[y] == FINAL_MODEL else TEXT_SECONDARY)
        ax.set_title(title, loc="left", fontsize=12, color=TEXT, fontweight="bold")
        ax.set_xlabel("Mean error (log scale)", color=TEXT_SECONDARY)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=TEXT_SECONDARY, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    fig.suptitle("Error when a whole blade geometry is held out of training (lower is better)",
                 x=0.01, ha="left", fontsize=13, color=TEXT)
    fig.tight_layout()
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=200, facecolor=SURFACE)
    print(f"\nFigure saved: {FIGURE_PATH.relative_to(REPO_ROOT)}")


def main():
    # The pure GPR's L-BFGS hits its iteration cap on this data; that is part of why it was replaced.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    X, y_stress, y_defl = get_features_and_targets(load_dataset())
    print("Comparing models (this takes a couple of minutes)...")
    results = evaluate(X, y_stress, y_defl)
    print_markdown(results)
    plot(results)


if __name__ == "__main__":
    main()
