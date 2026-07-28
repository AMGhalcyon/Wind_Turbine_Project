"""
Wind Turbine Blade Stress & Deflection Prediction Tool

@author: anish
"""

import os
import sys
import pickle
import warnings
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.utils import MATERIAL_COLS, PA_TO_MPA, M_TO_MM, NUMERIC_COLS

warnings.filterwarnings('ignore')


# Colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


# Training data ranges - warn user if they go outside these
TRAINING_RANGES = {
    "Blade_Length_m": (0.8, 1.6),
    "Root_Chord_mm": (150, 300),
    "Applied_Load_Pa": (500, 1500),
}

VALID_MATERIALS = ["Aluminium", "Fiberglass", "Carbon fiber"]

MODEL_DIR = Path(__file__).parent / "models"
STRESS_MODEL_PATH = MODEL_DIR / "Best_Stress_Model.pkl"
DEFLECTION_MODEL_PATH = MODEL_DIR / "Best_Deflection_Model.pkl"


def load_models():
    """Load the trained XGBoost models from disk."""
    if not STRESS_MODEL_PATH.exists():
        print(f"{Colors.RED}Error: Stress model not found at {STRESS_MODEL_PATH}{Colors.END}")
        print(f"{Colors.YELLOW}Please train and save your models first.{Colors.END}")
        sys.exit(1)

    if not DEFLECTION_MODEL_PATH.exists():
        print(f"{Colors.RED}Error: Deflection model not found at {DEFLECTION_MODEL_PATH}{Colors.END}")
        print(f"{Colors.YELLOW}Please train and save your models first.{Colors.END}")
        sys.exit(1)

    with open(STRESS_MODEL_PATH, 'rb') as f:
        stress_model = pickle.load(f)

    with open(DEFLECTION_MODEL_PATH, 'rb') as f:
        deflection_model = pickle.load(f)

    return stress_model, deflection_model


def validate_blade_length(value):
    try:
        val = float(value)
        if val <= 0:
            return None, "Blade length must be positive"
        return val, None
    except ValueError:
        return None, "Invalid number format"


def validate_chord_length(value):
    try:
        val = float(value)
        if val <= 0:
            return None, "Chord length must be positive"
        return val, None
    except ValueError:
        return None, "Invalid number format"


def validate_material(value):
    value = value.strip()

    # Check if material matches one of the valid ones (case insensitive)
    for valid_mat in VALID_MATERIALS:
        if value.lower() == valid_mat.lower():
            return valid_mat, None

    return None, f"Invalid material. Must be one of: {', '.join(VALID_MATERIALS)}"


def validate_load(value):
    try:
        val = float(value)
        if val <= 0:
            return None, "Applied load must be positive"
        return val, None
    except ValueError:
        return None, "Invalid number format"


def check_range_warnings(blade_length, chord_length, load):
    """Check if user input is outside the training data range."""
    warnings = []

    min_bl, max_bl = TRAINING_RANGES["Blade_Length_m"]
    if blade_length < min_bl or blade_length > max_bl:
        warnings.append(
            f"Blade Length = {blade_length:.2f} m is outside training range [{min_bl}, {max_bl}]"
        )

    min_chord, max_chord = TRAINING_RANGES["Root_Chord_mm"]
    if chord_length < min_chord or chord_length > max_chord:
        warnings.append(
            f"Chord Length = {chord_length:.0f} mm is outside training range [{min_chord}, {max_chord}]"
        )

    min_load, max_load = TRAINING_RANGES["Applied_Load_Pa"]
    if load < min_load or load > max_load:
        warnings.append(
            f"Applied Load = {load:.0f} Pa is outside training range [{min_load}, {max_load}]"
        )

    return warnings


def prepare_input_row(blade_length, chord_length, material, load):
    """Create a properly formatted input row for the model."""
    row = {
        "Blade_Length_m": blade_length,
        "Root_Chord_mm": chord_length,
        "Applied_Load_Pa": load
    }

    # Create one-hot encoded columns for material
    for mat_col in MATERIAL_COLS:
        expected_material = mat_col.replace("Material_", "")
        row[mat_col] = 1 if expected_material == material else 0

    return pd.DataFrame([row])


def make_prediction(stress_model, deflection_model, blade_length, chord_length, material, load):
    """Run predictions with both models."""
    input_df = prepare_input_row(blade_length, chord_length, material, load)

    # Models handle scaling internally via pipeline
    stress_pa = stress_model.predict(input_df)[0]
    deflection_m = deflection_model.predict(input_df)[0]

    # Convert to readable units (MPa and mm)
    stress_mpa = stress_pa / PA_TO_MPA
    deflection_mm = deflection_m * M_TO_MM

    return stress_mpa, deflection_mm


def print_header():
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'Wind Turbine Blade Prediction System':^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}\n")


def print_inputs(blade_length, chord_length, material, load):
    print(f"{Colors.BOLD}{Colors.BLUE}Input Parameters:{Colors.END}")
    print(f"{Colors.CYAN}├─ Blade Length  : {Colors.END}{blade_length:.2f} m")
    print(f"{Colors.CYAN}├─ Chord Length  : {Colors.END}{chord_length:.0f} mm")
    print(f"{Colors.CYAN}├─ Material      : {Colors.END}{material}")
    print(f"{Colors.CYAN}└─ Applied Load  : {Colors.END}{load:.0f} Pa")
    print()


def print_warnings(warnings):
    if warnings:
        print(f"{Colors.BOLD}{Colors.YELLOW}⚠ Engineering Warnings:{Colors.END}")
        print(f"{Colors.YELLOW}{'─' * 70}{Colors.END}")
        for warning in warnings:
            print(f"{Colors.YELLOW}  • {warning}{Colors.END}")
        print(f"{Colors.YELLOW}{'─' * 70}{Colors.END}")
        print(f"{Colors.YELLOW}  Predictions may be unreliable outside the training range.{Colors.END}")
        print()


def print_results(stress_mpa, deflection_mm):
    print(f"{Colors.BOLD}{Colors.GREEN}{'─' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}Prediction Results:{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'─' * 70}{Colors.END}")
    print()
    print(f"{Colors.BOLD}  Maximum Equivalent Stress:{Colors.END}")
    print(f"    {Colors.GREEN}{Colors.BOLD}{stress_mpa:.2f} MPa{Colors.END}")
    print()
    print(f"{Colors.BOLD}  Maximum Deflection:{Colors.END}")
    print(f"    {Colors.GREEN}{Colors.BOLD}{deflection_mm:.2f} mm{Colors.END}")
    print()
    print(f"{Colors.BOLD}{Colors.GREEN}{'─' * 70}{Colors.END}\n")


def print_footer():
    print(f"{Colors.CYAN}{'=' * 70}{Colors.END}\n")


def get_user_inputs():
    """Get blade parameters from user with validation."""
    print(f"{Colors.BOLD}Please enter the blade parameters:{Colors.END}\n")

    # Get blade length
    while True:
        blade_input = input(f"{Colors.CYAN}Blade Length (m): {Colors.END}").strip()
        blade_length, error = validate_blade_length(blade_input)
        if error:
            print(f"{Colors.RED}✗ {error}. Please try again.{Colors.END}")
        else:
            break

    # Get chord length
    while True:
        chord_input = input(f"{Colors.CYAN}Chord Length (mm): {Colors.END}").strip()
        chord_length, error = validate_chord_length(chord_input)
        if error:
            print(f"{Colors.RED}✗ {error}. Please try again.{Colors.END}")
        else:
            break

    # Get material
    print(f"\n{Colors.YELLOW}Available materials: {', '.join(VALID_MATERIALS)}{Colors.END}")
    while True:
        material_input = input(f"{Colors.CYAN}Material: {Colors.END}").strip()
        material, error = validate_material(material_input)
        if error:
            print(f"{Colors.RED}✗ {error}{Colors.END}")
        else:
            break

    # Get applied load
    while True:
        load_input = input(f"{Colors.CYAN}Applied Load (Pa): {Colors.END}").strip()
        load, error = validate_load(load_input)
        if error:
            print(f"{Colors.RED}✗ {error}. Please try again.{Colors.END}")
        else:
            break

    print()
    return blade_length, chord_length, material, load


def main():
    try:
        print_header()

        print(f"{Colors.YELLOW}Loading trained models...{Colors.END}")
        stress_model, deflection_model = load_models()
        print(f"{Colors.GREEN}✓ Models loaded successfully!{Colors.END}\n")

        blade_length, chord_length, material, load = get_user_inputs()
        print_inputs(blade_length, chord_length, material, load)

        # Warn if inputs are outside training range
        warnings = check_range_warnings(blade_length, chord_length, load)
        if warnings:
            print_warnings(warnings)

        print(f"{Colors.YELLOW}Computing predictions...{Colors.END}\n")
        stress_mpa, deflection_mm = make_prediction(
            stress_model, deflection_model,
            blade_length, chord_length, material, load
        )

        print_results(stress_mpa, deflection_mm)
        print_footer()

        # Option to run again
        another = input(f"{Colors.CYAN}Make another prediction? (y/n): {Colors.END}").strip().lower()
        if another == 'y':
            print("\n" * 2)
            main()
        else:
            print(f"\n{Colors.GREEN}Thank you for using the Wind Turbine Blade Prediction System!{Colors.END}\n")

    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Prediction cancelled by user.{Colors.END}\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}An unexpected error occurred:{Colors.END}")
        print(f"{Colors.RED}{str(e)}{Colors.END}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
