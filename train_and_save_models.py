"""
Train Gaussian Process Regression models and save them for the prediction application.

Uses train_final_models() which trains on the complete dataset with
pipelines that include preprocessing (StandardScaler + GPR).

@author: anish
"""

import pickle
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing import load_dataset, get_features_and_targets
from src.training import train_final_models
from src.utils import ensure_dir

# ANSI colors for pretty output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    END = '\033[0m'


def find_dataset():
    """Find the dataset CSV file."""
    # Try common locations
    possible_paths = [
        Path("Dataset/simulation_matrix_updated_info.csv"),
        Path("../Dataset/simulation_matrix_updated_info.csv"),
        Path("Python/../Dataset/simulation_matrix_updated_info.csv"),
    ]

    for path in possible_paths:
        if path.exists():
            return str(path)

    return None


def main():
    """Train and save the final GPR models."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'Training Final GPR Models':^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 70}{Colors.END}\n")

    # Find dataset
    print(f"{Colors.YELLOW}[1/4] Locating dataset...{Colors.END}")
    dataset_path = find_dataset()
    if not dataset_path:
        print(f"{Colors.RED}Error: Could not find simulation_matrix_updated_info.csv{Colors.END}")
        print(f"{Colors.YELLOW}Please ensure the Dataset directory exists with the CSV file{Colors.END}")
        sys.exit(1)
    print(f"{Colors.GREEN}      [OK] Found: {dataset_path}{Colors.END}\n")

    # Load and prepare data
    print(f"{Colors.YELLOW}[2/4] Loading and preparing data...{Colors.END}")
    data = load_dataset(dataset_path)
    X_encoded, y_stress, y_deflection = get_features_and_targets(data)
    print(f"{Colors.GREEN}      [OK] Loaded {len(data)} samples{Colors.END}")
    print(f"{Colors.GREEN}      [OK] Features: {X_encoded.shape[1]} columns (with one-hot encoding){Colors.END}\n")

    # Train final models (pipelines with StandardScaler + GPR)
    print(f"{Colors.YELLOW}[3/4] Training GPR models with preprocessing pipelines...{Colors.END}")
    print(f"      Note: Anisotropic RBF kernel + log-transform")
    print(f"      (This may take a few minutes due to optimizer restarts...)\n")

    stress_model, deflection_model = train_final_models(X_encoded, y_stress, y_deflection, model_type="gaussian_process")

    print(f"{Colors.GREEN}      [OK] Stress model trained (Pipeline: StandardScaler + GPR){Colors.END}")
    print(f"{Colors.GREEN}      [OK] Deflection model trained (Pipeline: StandardScaler + GPR){Colors.END}\n")

    # Save models
    print(f"{Colors.YELLOW}[4/4] Saving models...{Colors.END}")
    model_dir = Path(__file__).parent / "models"
    ensure_dir(str(model_dir))

    stress_path = model_dir / "Best_Stress_Model.pkl"
    deflection_path = model_dir / "Best_Deflection_Model.pkl"

    with open(stress_path, 'wb') as f:
        pickle.dump(stress_model, f)
    print(f"{Colors.GREEN}      [OK] {stress_path}{Colors.END}")

    with open(deflection_path, 'wb') as f:
        pickle.dump(deflection_model, f)
    print(f"{Colors.GREEN}      [OK] {deflection_path}{Colors.END}\n")

    print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}Training Complete!{Colors.END}")
    print(f"{Colors.BOLD}{Colors.GREEN}{'=' * 70}{Colors.END}\n")
    print(f"{Colors.CYAN}Models saved to: {model_dir}{Colors.END}")
    print(f"{Colors.CYAN}You can now run: python predict.py{Colors.END}\n")


if __name__ == "__main__":
    main()
