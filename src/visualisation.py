"""
@author: anish
"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_actual_vs_predicted(y_true, y_pred, xlabel, ylabel, title, color="steelblue", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=70, alpha=0.8, edgecolor="k", color=color)
    lims = [min(min(y_true), min(y_pred)), max(max(y_true), max(y_pred))]
    ax.plot(lims, lims, "r--", label="Perfect prediction")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_error_distribution(errors, xlabel, title, color="steelblue", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(errors, kde=True, bins=12, color=color, ax=ax)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_residuals(y_pred, residuals, xlabel, ylabel, title, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(y_pred, residuals, s=70, alpha=0.8, edgecolor="k")
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_feature_importance(importance_series, title, color="indianred", ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    importance_series.plot(kind="barh", ax=ax, color=color)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_cv_boxplot(r2_data, labels, colors, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    bp = ax.boxplot(r2_data, tick_labels=labels, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel("Cross-Validation R\u00b2 Score")
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_learning_curve(sizes, train_mean, train_std, val_mean, val_std, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(sizes, train_mean, "o-", color="darkorange", label="Training Score")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="darkorange")
    ax.plot(sizes, val_mean, "o-", color="steelblue", label="Validation Score")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="steelblue")
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("R\u00b2 Score")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return ax


def plot_parameter_sweep(x_values, stress_vals, defl_vals, xlabel, title_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(x_values, stress_vals, "o-", color="indianred")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Max Stress (MPa)")
    axes[0].set_title(f"{title_prefix} vs Stress")

    axes[1].plot(x_values, defl_vals, "o-", color="steelblue")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Max Deflection (mm)")
    axes[1].set_title(f"{title_prefix} vs Deflection")
    plt.tight_layout()
    return axes