"""
@author: anish
"""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from src.utils import NUMERIC_COLS, RANDOM_STATE


def build_preprocessor(numeric_cols=NUMERIC_COLS) -> ColumnTransformer:
    # scales numeric cols, passes one-hot Material cols through
    return ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_cols)],
        remainder="passthrough",
    )


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


def train_gaussian_process(X_train, y_train, kernel=None, alpha=1e-8,
                           n_restarts_optimizer=15, random_state=RANDOM_STATE) -> TransformedTargetRegressor:
    if kernel is None:
        # Anisotropic RBF kernel (ARD) provides the best performance for stress and deflection
        n_features = X_train.shape[1]
        kernel = ConstantKernel(1.0, (1e-5, 1e7)) * RBF(
            length_scale=[1.0] * n_features,
            length_scale_bounds=(1e-2, 1e3)
        )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        n_restarts_optimizer=n_restarts_optimizer,
        normalize_y=True,
        random_state=random_state
    )
    model = TransformedTargetRegressor(
        regressor=gpr,
        func=np.log,
        inverse_func=np.exp
    )
    model.fit(X_train, y_train)
    return model


def build_pipeline(model, numeric_cols=NUMERIC_COLS) -> Pipeline:
    # scaling inside the pipeline so CV folds don't leak
    if isinstance(model, GaussianProcessRegressor):
        model = TransformedTargetRegressor(regressor=model, func=np.log, inverse_func=np.exp)
    return Pipeline([("prep", build_preprocessor(numeric_cols)), ("model", model)])


def train_final_models(X_encoded, y_stress, y_deflection, model_type="xgboost"):
    # train final models on complete dataset for design exploration
    if model_type == "xgboost":
        stress_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE)
        defl_model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE)
    elif model_type == "gaussian_process":
        # Create default scale-matching kernels for the whole encoded dataset
        # We need the number of features after numeric scaling and categorical passthrough
        n_features = X_encoded.shape[1]
        kernel_stress = ConstantKernel(1.0, (1e-5, 1e7)) * RBF(
            length_scale=[1.0] * n_features,
            length_scale_bounds=(1e-2, 1e3)
        )
        kernel_defl = ConstantKernel(1.0, (1e-5, 1e7)) * RBF(
            length_scale=[1.0] * n_features,
            length_scale_bounds=(1e-2, 1e3)
        )
        stress_gpr = GaussianProcessRegressor(kernel=kernel_stress, alpha=1e-8, n_restarts_optimizer=15, normalize_y=True, random_state=RANDOM_STATE)
        defl_gpr = GaussianProcessRegressor(kernel=kernel_defl, alpha=1e-8, n_restarts_optimizer=15, normalize_y=True, random_state=RANDOM_STATE)
        stress_model = TransformedTargetRegressor(regressor=stress_gpr, func=np.log, inverse_func=np.exp)
        defl_model = TransformedTargetRegressor(regressor=defl_gpr, func=np.log, inverse_func=np.exp)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    stress_pipeline = build_pipeline(stress_model)
    deflection_pipeline = build_pipeline(defl_model)

    stress_pipeline.fit(X_encoded, y_stress)
    deflection_pipeline.fit(X_encoded, y_deflection)
    return stress_pipeline, deflection_pipeline