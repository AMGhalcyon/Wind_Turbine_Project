"""
Model builders: tree baselines, the pure GPR, the Euler-Bernoulli power law,
and the physics-informed GP (power law + GP on what the power law misses).

@author: anish
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils import MATERIAL_COLS, NUMERIC_COLS, RANDOM_STATE

N_ENCODED_FEATURES = len(NUMERIC_COLS) + len(MATERIAL_COLS)
RESIDUAL_GEOMETRY_COLS = ["Blade_Length_m", "Root_Chord_mm"]


def build_preprocessor(numeric_cols=NUMERIC_COLS) -> ColumnTransformer:
    # scales numeric cols, passes one-hot Material cols through
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_cols)],
        remainder="passthrough",
    )


def make_gpr(n_features: int = N_ENCODED_FEATURES, alpha: float = 1e-8,
             n_restarts_optimizer: int = 15, random_state: int = RANDOM_STATE) -> GaussianProcessRegressor:
    """Pure GPR with an anisotropic (ARD) RBF kernel - one length scale per input."""
    kernel = ConstantKernel(1.0, (1e-5, 1e7)) * RBF(length_scale=[1.0] * n_features,
                                                    length_scale_bounds=(1e-2, 1e3))
    return GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer,
                                    normalize_y=True, random_state=random_state)


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None,
                         min_samples_split=2, random_state=RANDOM_STATE) -> RandomForestRegressor:
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_split=min_samples_split, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, n_estimators=100, max_depth=4,
                   learning_rate=0.1, random_state=RANDOM_STATE) -> XGBRegressor:
    model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                          learning_rate=learning_rate, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_gaussian_process(X_train, y_train, kernel=None, alpha=1e-8, n_restarts_optimizer=15,
                           random_state=RANDOM_STATE) -> TransformedTargetRegressor:
    gpr = make_gpr(X_train.shape[1], alpha, n_restarts_optimizer, random_state)
    if kernel is not None:
        gpr.set_params(kernel=kernel)
    model = TransformedTargetRegressor(regressor=gpr, func=np.log, inverse_func=np.exp)
    model.fit(X_train, y_train)
    return model


def build_pipeline(model, numeric_cols=NUMERIC_COLS) -> Pipeline:
    # scaling inside the pipeline so CV folds don't leak
    if isinstance(model, (PowerLawModel, PhysicsInformedGP)):
        return model  # need raw, named columns (they take logs and scale internally)
    if isinstance(model, GaussianProcessRegressor):
        model = TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)
    return Pipeline([("prep", build_preprocessor(numeric_cols)), ("model", model)])


def _check_positive(y) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if np.any(y <= 0):
        raise ValueError("targets must be strictly positive (models work in log space)")
    return y


def _power_law_design(X: pd.DataFrame) -> np.ndarray:
    # log y = a + b*log(L) + c*log(chord) + d*log(P) + material offsets (Aluminium = baseline)
    logs = np.log(X[NUMERIC_COLS].to_numpy(dtype=float))
    materials = X[MATERIAL_COLS[1:]].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(X)), logs, materials])


class PowerLawModel(RegressorMixin, BaseEstimator):
    """Euler-Bernoulli scaling law fitted by least squares in log space (6 parameters).

    Expects a DataFrame with ENCODED_COLS. `exponents_` holds the fitted powers of
    blade length, root chord and load - theory says (2, -2, 1) for stress, (4, -3, 1) for deflection.
    """

    def fit(self, X: pd.DataFrame, y) -> "PowerLawModel":
        log_y = np.log(_check_positive(y))
        self.coef_ = np.linalg.lstsq(_power_law_design(X), log_y, rcond=None)[0]
        self.exponents_ = dict(zip(NUMERIC_COLS, self.coef_[1:4]))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.exp(_power_law_design(X) @ self.coef_)


class PhysicsInformedGP(RegressorMixin, BaseEstimator):
    """Power law does the heavy lifting; a GP learns the small log-residual it misses.

    Outside the training envelope the GP correction decays to zero, so predictions
    fall back to the physics law instead of drifting.
    """

    def __init__(self, n_restarts_optimizer: int = 15, random_state: int = RANDOM_STATE):
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y) -> "PhysicsInformedGP":
        y = _check_positive(y)
        self.power_law_ = PowerLawModel().fit(X, y)
        residual = np.log(y) - np.log(self.power_law_.predict(X))
        # Load is left out on purpose: linear static FEA makes every response exactly proportional
        # to load, which the power law already captures (fitted exponent 1.00) - nothing left to learn.
        prep = ColumnTransformer([("scale", StandardScaler(), RESIDUAL_GEOMETRY_COLS),
                                  ("materials", "passthrough", MATERIAL_COLS)])
        n_inputs = len(RESIDUAL_GEOMETRY_COLS) + len(MATERIAL_COLS)
        # Lower bound 0.5 (standardised units ~ half a grid step): 3 levels per variable cannot
        # resolve anything finer, and without it the GP memorises FEA/mesh noise between levels.
        # Wide upper bound lets the GP switch off an input it finds irrelevant.
        kernel = ConstantKernel(1e-2, (1e-8, 1e2)) * RBF(length_scale=[1.0] * n_inputs,
                                                         length_scale_bounds=(0.5, 1e6))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=self.n_restarts_optimizer,
                                       random_state=self.random_state)
        self.residual_gp_ = Pipeline([("prep", prep), ("gp", gpr)]).fit(X, residual)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.power_law_.predict(X) * np.exp(self.residual_gp_.predict(X))


def train_final_models(X_encoded, y_stress, y_deflection, model_type="xgboost"):
    """Train both targets on the full dataset. model_type: xgboost | gaussian_process | physics_gp."""
    if model_type == "physics_gp":
        return PhysicsInformedGP().fit(X_encoded, y_stress), PhysicsInformedGP().fit(X_encoded, y_deflection)
    if model_type == "xgboost":
        make = lambda: XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE)
    elif model_type == "gaussian_process":
        make = lambda: make_gpr(X_encoded.shape[1])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    stress_pipeline = build_pipeline(make()).fit(X_encoded, y_stress)
    deflection_pipeline = build_pipeline(make()).fit(X_encoded, y_deflection)
    return stress_pipeline, deflection_pipeline
