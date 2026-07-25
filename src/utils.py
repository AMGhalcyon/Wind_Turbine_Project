"""
@author: anish
"""

import os

DATASET_PATH = "..\\Dataset\\simulation_matrix_updated_info.csv"
RANDOM_STATE = 42

FEATURE_COLS = ["Blade_Length_m", "Root_Chord_mm", "Material", "Applied_Load_Pa"]
NUMERIC_COLS = ["Blade_Length_m", "Root_Chord_mm", "Applied_Load_Pa"]
TARGET_STRESS = "Max_Equiv_Stress_Pa"
TARGET_DEFLECTION = "Max_Deformation_m"
MATERIAL_COLS = ["Material_Aluminium", "Material_Carbon fiber", "Material_Fiberglass"]

PA_TO_MPA = 1e6
M_TO_MM = 1000


def resolve_dataset_path(path: str = DATASET_PATH) -> str:
    # Windows path as authored; convert for Linux sandbox execution
    if os.name == "nt":
        return path
    return path.replace("\\", "/")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)