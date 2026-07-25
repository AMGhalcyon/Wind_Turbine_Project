"""
@author: anish
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.utils import (DATASET_PATH, FEATURE_COLS, NUMERIC_COLS, RANDOM_STATE,
    TARGET_DEFLECTION, TARGET_STRESS, resolve_dataset_path, ensure_dir,)


def load_dataset(path: str = DATASET_PATH) -> pd.DataFrame:
    data = pd.read_csv(resolve_dataset_path(path))
    # stress column has a stray space in some exports
    data[TARGET_STRESS] = data[TARGET_STRESS].astype(str).str.replace(" ", "").astype(float)
    return data


def encode_materials(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X, columns=["Material"], prefix="Material")


def get_features_and_targets(data: pd.DataFrame):
    X = data[FEATURE_COLS].copy()
    y_stress = data[TARGET_STRESS].copy()
    y_deflection = data[TARGET_DEFLECTION].copy()
    return encode_materials(X), y_stress, y_deflection


def split_dataset(X_encoded, y, test_size=0.2, random_state=RANDOM_STATE):
    return train_test_split(X_encoded, y, test_size=test_size, random_state=random_state)


def scale_numeric_features(X_train_raw, X_test_raw, numeric_cols=NUMERIC_COLS):
    scaler = StandardScaler()
    X_train_scaled = X_train_raw.copy()
    X_test_scaled = X_test_raw.copy()
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_raw[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test_raw[numeric_cols])
    return X_train_scaled, X_test_scaled, scaler


def prepare_ml_ready_data(path: str = DATASET_PATH, save_dir: str = None):
    data = load_dataset(path)
    X_encoded, y_stress, y_deflection = get_features_and_targets(data)

    X_train_s, X_test_s, y_train_s, y_test_s = split_dataset(X_encoded, y_stress)
    X_train_d, X_test_d, y_train_d, y_test_d = split_dataset(X_encoded, y_deflection)

    X_train_s_scaled, X_test_s_scaled, scaler_s = scale_numeric_features(X_train_s, X_test_s)
    X_train_d_scaled, X_test_d_scaled, scaler_d = scale_numeric_features(X_train_d, X_test_d)

    result = {
        "stress": {"X_train": X_train_s_scaled, "X_test": X_test_s_scaled,
                   "y_train": y_train_s, "y_test": y_test_s, "scaler": scaler_s},
        "deflection": {"X_train": X_train_d_scaled, "X_test": X_test_d_scaled,
                       "y_train": y_train_d, "y_test": y_test_d, "scaler": scaler_d},
    }
    if save_dir:
        save_ml_ready_data(result, save_dir)
    return result


def save_ml_ready_data(result: dict, save_dir: str = "ml_ready_data") -> None:
    ensure_dir(save_dir)
    for target, d in result.items():
        d["X_train"].to_csv(f"{save_dir}/X_train_{target}.csv", index=False)
        d["X_test"].to_csv(f"{save_dir}/X_test_{target}.csv", index=False)
        d["y_train"].to_csv(f"{save_dir}/y_train_{target}.csv", index=False)
        d["y_test"].to_csv(f"{save_dir}/y_test_{target}.csv", index=False)