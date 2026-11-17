"""
Checks that fail if the physics, the model, or the CLI input handling breaks.

Run:  python tests/test_models.py      (or `pytest tests/` if you have pytest)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from predict import check_range_warnings, validate_material, validate_positive
from src.data_processing import get_features_and_targets, load_dataset
from src.prediction import make_row, predict_new_sample
from src.training import PhysicsInformedGP, PowerLawModel
from src.utils import ENCODED_COLS, OFF_GRID_ANSYS, OFF_GRID_CASE

X, Y_STRESS, Y_DEFL = get_features_and_targets(load_dataset())


def test_dataset_is_complete_factorial():
    assert len(X) == 81
    assert X[ENCODED_COLS[:3]].drop_duplicates().shape[0] == 27  # 3 lengths x 3 chords x 3 loads
    assert (Y_STRESS > 0).all() and (Y_DEFL > 0).all()


def test_power_law_recovers_beam_theory_exponents():
    stress = PowerLawModel().fit(X, Y_STRESS).exponents_
    defl = PowerLawModel().fit(X, Y_DEFL).exponents_
    assert np.allclose(list(stress.values()), [2, -2, 1], atol=0.1), stress
    assert np.allclose(list(defl.values()), [4, -3, 1], atol=0.15), defl


def test_physics_gp_matches_off_grid_ansys_run():
    stress_model = PhysicsInformedGP(n_restarts_optimizer=2).fit(X, Y_STRESS)
    defl_model = PhysicsInformedGP(n_restarts_optimizer=2).fit(X, Y_DEFL)
    result = predict_new_sample(stress_model, defl_model, **OFF_GRID_CASE)
    stress_err = abs(result["stress_MPa"] / OFF_GRID_ANSYS["stress_MPa"] - 1)
    defl_err = abs(result["deflection_mm"] / OFF_GRID_ANSYS["deflection_mm"] - 1)
    assert stress_err < 0.02 and defl_err < 0.02, (result, OFF_GRID_ANSYS)


def test_physics_gp_rejects_non_positive_targets():
    try:
        PhysicsInformedGP(n_restarts_optimizer=0).fit(X, np.zeros(len(X)))
    except ValueError:
        return
    raise AssertionError("expected ValueError for zero targets")


def test_make_row_one_hot_and_unknown_material():
    row = make_row(1.2, 200, "Carbon fiber", 1000)
    assert [row[c] for c in ENCODED_COLS[3:]] == [0, 1, 0]
    assert list(pd.DataFrame([row])[ENCODED_COLS].columns) == ENCODED_COLS
    try:
        make_row(1.2, 200, "Steel", 1000)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown material")


def test_cli_validation():
    assert validate_positive("1.3", "x") == (1.3, None)
    for bad in ("-1", "0", "abc", "nan", "inf"):
        assert validate_positive(bad, "x")[0] is None, bad
    assert validate_material("  carbon FIBER ") == ("Carbon fiber", None)
    assert validate_material("steel")[0] is None


def test_range_warnings():
    assert check_range_warnings(1.3, 180, 1200) == []
    assert len(check_range_warnings(2.0, 100, 2000)) == 3


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n{len(tests)} tests passed")
