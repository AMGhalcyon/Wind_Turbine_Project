"""
Train the final physics-informed models on all 81 simulations and save them for predict.py.

Each model = Euler-Bernoulli power law fitted in log space + a Gaussian Process that
learns the small residual the power law misses. See compare_models.py for why.

@author: anish
"""

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import get_features_and_targets, load_dataset
from src.training import train_final_models
from src.utils import (DATASET_PATH, DEFLECTION_MODEL_PATH, MODEL_DIR, STRESS_MODEL_PATH,
                       Colors, ensure_dir)


def step(text: str):
    print(f"{Colors.YELLOW}{text}{Colors.END}")


def ok(text: str):
    print(f"{Colors.GREEN}      [OK] {text}{Colors.END}")


def main():
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}\n{'Training Final Physics-Informed Models':^70}\n{'=' * 70}{Colors.END}\n")

    step("[1/3] Loading dataset...")
    if not DATASET_PATH.exists():
        print(f"{Colors.RED}Error: dataset not found at {DATASET_PATH}{Colors.END}")
        sys.exit(1)
    data = load_dataset(DATASET_PATH)
    X_encoded, y_stress, y_deflection = get_features_and_targets(data)
    ok(f"{len(data)} simulations, {X_encoded.shape[1]} encoded features\n")

    step("[2/3] Fitting power law + residual Gaussian Process (takes ~1 minute)...")
    stress_model, deflection_model = train_final_models(X_encoded, y_stress, y_deflection, model_type="physics_gp")
    for name, model in (("Stress", stress_model), ("Deflection", deflection_model)):
        exps = model.power_law_.exponents_
        ok(f"{name}: L^{exps['Blade_Length_m']:.2f} · c^{exps['Root_Chord_mm']:.2f} · P^{exps['Applied_Load_Pa']:.2f}")
    print()

    step("[3/3] Saving models...")
    ensure_dir(MODEL_DIR)
    for path, model in ((STRESS_MODEL_PATH, stress_model), (DEFLECTION_MODEL_PATH, deflection_model)):
        with open(path, "wb") as f:
            pickle.dump(model, f)
        ok(str(path))

    print(f"\n{Colors.BOLD}{Colors.GREEN}Training complete.{Colors.END} Run: python predict.py\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
