"""
@author: anish
"""
import pandas as pd

from src.utils import MATERIAL_COLS, PA_TO_MPA, M_TO_MM


def make_row(blade_length, chord, material, load) -> dict:
    row = {"Blade_Length_m": blade_length, "Root_Chord_mm": chord, "Applied_Load_Pa": load}
    for m in MATERIAL_COLS:
        row[m] = 1 if m == f"Material_{material}" else 0
    return row


def predict_new_sample(stress_model, deflection_model, feature_cols,
                        blade_length, chord, material, load) -> dict:
    row = make_row(blade_length, chord, material, load)
    df = pd.DataFrame([row])[feature_cols]
    stress_pa = stress_model.predict(df)[0]
    defl_m = deflection_model.predict(df)[0]
    return {
        "stress_MPa": stress_pa / PA_TO_MPA,
        "deflection_mm": defl_m * M_TO_MM,
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
