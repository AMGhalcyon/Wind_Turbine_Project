"""
Turn blade parameters into model inputs and run predictions / parameter sweeps.

@author: anish
"""
import pandas as pd

from src.utils import ENCODED_COLS, MATERIAL_COLS, M_TO_MM, PA_TO_MPA


def make_row(blade_length, chord, material, load) -> dict:
    material_col = f"Material_{material}"
    if material_col not in MATERIAL_COLS:
        raise ValueError(f"Unknown material {material!r}")
    row = {"Blade_Length_m": blade_length, "Root_Chord_mm": chord, "Applied_Load_Pa": load}
    for m in MATERIAL_COLS:
        row[m] = 1 if m == material_col else 0
    return row


def predict_new_sample(stress_model, deflection_model, blade_length, chord, material, load,
                       feature_cols=ENCODED_COLS) -> dict:
    df = pd.DataFrame([make_row(blade_length, chord, material, load)])[list(feature_cols)]
    return {
        "stress_MPa": stress_model.predict(df)[0] / PA_TO_MPA,
        "deflection_mm": deflection_model.predict(df)[0] * M_TO_MM,
    }


def sweep(stress_model, deflection_model, feature_cols, param_name, values, fixed):
    rows = []
    for v in values:
        params = dict(fixed)
        params[param_name] = v
        rows.append(make_row(**params))
    df = pd.DataFrame(rows)[feature_cols]
    return stress_model.predict(df), deflection_model.predict(df)


def ratio(arr):
    return max(arr) / min(arr) if min(arr) != 0 else float("nan")
