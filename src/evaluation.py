"""
@author: anish
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, learning_curve

from src.training import build_pipeline
from src.utils import RANDOM_STATE


def calculate_metrics(y_true, y_pred, unit_divisor=1.0) -> dict:
    y_true_u = np.asarray(y_true) / unit_divisor
    y_pred_u = np.asarray(y_pred) / unit_divisor
    return {
        "MAE": mean_absolute_error(y_true_u, y_pred_u),
        "RMSE": np.sqrt(mean_squared_error(y_true_u, y_pred_u)),
        "R2": r2_score(y_true_u, y_pred_u),
    }


def feature_importance(model, feature_names) -> pd.Series:
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)


def cross_validation(model, X, y, unit_divisor=1.0, n_splits=5, random_state=RANDOM_STATE) -> dict:
    pipe = build_pipeline(model)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scoring = {"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error", "R2": "r2"}
    cv_results = cross_validate(pipe, X, y, cv=kf, scoring=scoring, return_train_score=False)
    return {
        "MAE": -cv_results["test_MAE"] / unit_divisor,
        "RMSE": -cv_results["test_RMSE"] / unit_divisor,
        "R2": cv_results["test_R2"],
    }


def learning_curve_analysis(model, X, y, n_splits=5, train_sizes=np.linspace(0.2, 1.0, 8),
                             random_state=RANDOM_STATE) -> dict:
    pipe = build_pipeline(model)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    sizes, train_scores, val_scores = learning_curve(
        pipe, X, y, train_sizes=train_sizes, cv=kf, scoring="r2", shuffle=True, random_state=random_state
    )
    return {
        "sizes": sizes,
        "train_mean": train_scores.mean(axis=1), "train_std": train_scores.std(axis=1),
        "val_mean": val_scores.mean(axis=1), "val_std": val_scores.std(axis=1),
    }


def diagnose_fit(train_score, val_score, gap_threshold=0.15) -> str:
    gap = train_score - val_score
    if gap > gap_threshold:
        return "Overfitting"
    elif train_score < 0.85 and val_score < 0.85:
        return "Underfitting"
    return "Well-fitted"