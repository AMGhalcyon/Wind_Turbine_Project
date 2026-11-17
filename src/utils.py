"""
Shared constants: column names, unit conversions, paths, and the design envelope.

@author: anish
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = REPO_ROOT / "Dataset" / "simulation_matrix_updated_info.csv"
MODEL_DIR = REPO_ROOT / "models"
STRESS_MODEL_PATH = MODEL_DIR / "Best_Stress_Model.pkl"
DEFLECTION_MODEL_PATH = MODEL_DIR / "Best_Deflection_Model.pkl"

RANDOM_STATE = 42

FEATURE_COLS = ["Blade_Length_m", "Root_Chord_mm", "Material", "Applied_Load_Pa"]
NUMERIC_COLS = ["Blade_Length_m", "Root_Chord_mm", "Applied_Load_Pa"]
TARGET_STRESS = "Max_Equiv_Stress_Pa"
TARGET_DEFLECTION = "Max_Deformation_m"
MATERIAL_COLS = ["Material_Aluminium", "Material_Carbon fiber", "Material_Fiberglass"]
ENCODED_COLS = NUMERIC_COLS + MATERIAL_COLS  # column order produced by get_dummies
VALID_MATERIALS = ["Aluminium", "Fiberglass", "Carbon fiber"]

# Sampled design envelope - the models have no training support outside it
TRAINING_RANGES = {
    "Blade_Length_m": (0.8, 1.6),
    "Root_Chord_mm": (150, 300),
    "Applied_Load_Pa": (500, 1500),
}

# One extra ANSYS run at a point that is off-grid in three variables at once,
# used as the only truly independent accuracy check
OFF_GRID_CASE = {"blade_length": 1.3, "chord": 180, "material": "Carbon fiber", "load": 1200}
OFF_GRID_ANSYS = {"stress_MPa": 18.85, "deflection_mm": 13.67}

PA_TO_MPA = 1e6
M_TO_MM = 1000


class Colors:
    """ANSI colour codes for the CLI scripts."""
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ensure_dir(path) -> None:
    os.makedirs(path, exist_ok=True)
