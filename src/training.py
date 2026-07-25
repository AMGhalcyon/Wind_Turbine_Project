"""
@author: anish
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
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


def build_pipeline(model, numeric_cols=NUMERIC_COLS) -> Pipeline:
    # scaling inside the pipeline so CV folds don't leak
    return Pipeline([("prep", build_preprocessor(numeric_cols)), ("model", model)])


def train_final_models(X_encoded, y_stress, y_deflection):
    # per-target model selection: XGBoost for stress, RF for deflection
    stress_pipeline = build_pipeline(
        XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE)
    )
    deflection_pipeline = build_pipeline(
        RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
    )
    stress_pipeline.fit(X_encoded, y_stress)
    deflection_pipeline.fit(X_encoded, y_deflection)
    return stress_pipeline, deflection_pipeline