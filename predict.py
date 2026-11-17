"""
Wind Turbine Blade Stress & Deflection Prediction Tool (interactive CLI).

Loads the models written by train_and_save_models.py. Only load pickles you trained
yourself - unpickling runs code.

@author: anish
"""

import math
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.prediction import predict_new_sample
from src.utils import (DEFLECTION_MODEL_PATH, STRESS_MODEL_PATH, TRAINING_RANGES,
                       VALID_MATERIALS, Colors)

RULE = "=" * 70
THIN = "─" * 70
RANGE_LABELS = {  # input name -> (display label, unit, format)
    "Blade_Length_m": ("Blade Length", "m", ".2f"),
    "Root_Chord_mm": ("Chord Length", "mm", ".0f"),
    "Applied_Load_Pa": ("Applied Load", "Pa", ".0f"),
}


def load_models():
    """Load the trained stress and deflection models from disk."""
    for path in (STRESS_MODEL_PATH, DEFLECTION_MODEL_PATH):
        if not path.exists():
            print(f"{Colors.RED}Error: model not found at {path}{Colors.END}")
            print(f"{Colors.YELLOW}Run `python train_and_save_models.py` first.{Colors.END}")
            sys.exit(1)
    with open(STRESS_MODEL_PATH, "rb") as f:
        stress_model = pickle.load(f)
    with open(DEFLECTION_MODEL_PATH, "rb") as f:
        deflection_model = pickle.load(f)
    return stress_model, deflection_model


def validate_positive(value: str, label: str):
    """Parse a strictly positive, finite number. Returns (value, error)."""
    try:
        val = float(value)
    except ValueError:
        return None, "Invalid number format"
    if not math.isfinite(val) or val <= 0:
        return None, f"{label} must be a positive number"
    return val, None


def validate_material(value: str):
    """Case-insensitive match against the simulated materials. Returns (material, error)."""
    for material in VALID_MATERIALS:
        if value.strip().lower() == material.lower():
            return material, None
    return None, f"Invalid material. Must be one of: {', '.join(VALID_MATERIALS)}"


def check_range_warnings(blade_length: float, chord_length: float, load: float) -> list:
    """One warning per input that falls outside the simulated design envelope."""
    values = {"Blade_Length_m": blade_length, "Root_Chord_mm": chord_length, "Applied_Load_Pa": load}
    warnings = []
    for key, value in values.items():
        low, high = TRAINING_RANGES[key]
        if not low <= value <= high:
            label, unit, fmt = RANGE_LABELS[key]
            warnings.append(f"{label} = {value:{fmt}} {unit} is outside the simulated range [{low}, {high}]")
    return warnings


def ask(prompt: str, validator):
    """Re-prompt until the validator accepts the input."""
    while True:
        value, error = validator(input(f"{Colors.CYAN}{prompt}: {Colors.END}").strip())
        if error is None:
            return value
        print(f"{Colors.RED}✗ {error}. Please try again.{Colors.END}")


def get_user_inputs():
    print(f"{Colors.BOLD}Please enter the blade parameters:{Colors.END}\n")
    blade_length = ask("Blade Length (m)", lambda v: validate_positive(v, "Blade length"))
    chord_length = ask("Chord Length (mm)", lambda v: validate_positive(v, "Chord length"))
    print(f"\n{Colors.YELLOW}Available materials: {', '.join(VALID_MATERIALS)}{Colors.END}")
    material = ask("Material", validate_material)
    load = ask("Applied Load (Pa)", lambda v: validate_positive(v, "Applied load"))
    print()
    return blade_length, chord_length, material, load


def print_inputs(blade_length, chord_length, material, load):
    print(f"{Colors.BOLD}{Colors.BLUE}Input Parameters:{Colors.END}")
    print(f"{Colors.CYAN}├─ Blade Length  : {Colors.END}{blade_length:.2f} m")
    print(f"{Colors.CYAN}├─ Chord Length  : {Colors.END}{chord_length:.0f} mm")
    print(f"{Colors.CYAN}├─ Material      : {Colors.END}{material}")
    print(f"{Colors.CYAN}└─ Applied Load  : {Colors.END}{load:.0f} Pa\n")


def print_warnings(warnings: list):
    print(f"{Colors.BOLD}{Colors.YELLOW}⚠ Engineering Warnings:{Colors.END}")
    print(f"{Colors.YELLOW}{THIN}{Colors.END}")
    for warning in warnings:
        print(f"{Colors.YELLOW}  • {warning}{Colors.END}")
    print(f"{Colors.YELLOW}{THIN}{Colors.END}")
    print(f"{Colors.YELLOW}  No simulations back this region: treat the prediction as an")
    print(f"  unverified extrapolation.{Colors.END}\n")


def print_results(stress_mpa: float, deflection_mm: float):
    print(f"{Colors.BOLD}{Colors.GREEN}{THIN}\nPrediction Results:\n{THIN}{Colors.END}\n")
    print(f"{Colors.BOLD}  Maximum Equivalent Stress:{Colors.END}")
    print(f"    {Colors.GREEN}{Colors.BOLD}{stress_mpa:.2f} MPa{Colors.END}\n")
    print(f"{Colors.BOLD}  Maximum Deflection:{Colors.END}")
    print(f"    {Colors.GREEN}{Colors.BOLD}{deflection_mm:.2f} mm{Colors.END}\n")
    print(f"{Colors.BOLD}{Colors.GREEN}{THIN}{Colors.END}\n")


def run_once(stress_model, deflection_model):
    blade_length, chord_length, material, load = get_user_inputs()
    print_inputs(blade_length, chord_length, material, load)
    warnings = check_range_warnings(blade_length, chord_length, load)
    if warnings:
        print_warnings(warnings)
    print(f"{Colors.YELLOW}Computing predictions...{Colors.END}\n")
    result = predict_new_sample(stress_model, deflection_model, blade_length, chord_length, material, load)
    if not all(math.isfinite(v) for v in result.values()):
        print(f"{Colors.RED}✗ Inputs are too extreme for a finite prediction.{Colors.END}\n")
        return
    print_results(result["stress_MPa"], result["deflection_mm"])
    print(f"{Colors.CYAN}{RULE}{Colors.END}\n")


def main():
    if hasattr(sys.stdout, "reconfigure"):  # box-drawing chars crash cp1252 consoles otherwise
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"\n{Colors.BOLD}{Colors.CYAN}{RULE}\n{'Wind Turbine Blade Prediction System':^70}\n{RULE}{Colors.END}\n")
    try:
        print(f"{Colors.YELLOW}Loading trained models...{Colors.END}")
        stress_model, deflection_model = load_models()
        print(f"{Colors.GREEN}✓ Models loaded successfully!{Colors.END}\n")
        while True:
            run_once(stress_model, deflection_model)
            if input(f"{Colors.CYAN}Make another prediction? (y/n): {Colors.END}").strip().lower() != "y":
                break
            print("\n")
        print(f"\n{Colors.GREEN}Thank you for using the Wind Turbine Blade Prediction System!{Colors.END}\n")
    except (KeyboardInterrupt, EOFError):
        print(f"\n\n{Colors.YELLOW}Prediction cancelled by user.{Colors.END}\n")


if __name__ == "__main__":
    main()
