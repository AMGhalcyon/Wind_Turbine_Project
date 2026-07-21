# Week 7 Notes


## Why XGBoost?

Random Forest builds many independent trees and averages them; XGBoost builds trees sequentially, each correcting the previous ones' errors (gradient boosting). Testing a second algorithm under the same preprocessing pipeline is good practice - it confirms whether "the model" or "the specific algorithm" is what drives results, and avoids concluding an approach is best after testing only one candidate. XGBoost specifically is widely used for tabular/engineering data because it captures nonlinear relationships and feature interactions natively (like RF), but often achieves tighter fits on smooth, low-noise numerical data - relevant here since FEA outputs are deterministic, not noisy measurements.

## Model Comparison Results

| Metric | RF Stress | XGB Stress | RF Deflection | XGB Deflection |
|---|---|---|---|---|
| MAE | 1.747 MPa | **0.072 MPa** | 11.299 mm | 11.5119 mm |
| RMSE | 2.234 MPa | **0.086 MPa** | 20.172 mm | 22.8623 mm |
| R² | 0.9388 | **0.9999** | 0.696 | 0.6097 |

**Best performer:** XGBoost for Stress (large, decisive improvement). Random Forest for Deflection (XGBoost is slightly worse on R²/RMSE, marginally better on MAE only).

## Discussion

- **Stress:** XGBoost improved dramatically (R² 0.939 -> 0.9999). Train vs. test R² checked to rule out overfitting: 0.99999 train / 0.9999 test - almost no gap, so this is genuine generalization, not memorization. Stress is a smooth, material-independent, low-noise function of geometry and load, which suits boosting's precision well.
- **Deflection:** XGBoost got *worse* (R² 0.696 -> 0.6097), and its train/test gap (0.999 train vs 0.6097 test) is much larger than RF's own gap on the same target - i.e., XGBoost is overfitting deflection specifically.
- **Consistent with dataset size (81 simulations)?** Yes, directly. With only 64 training rows, XGBoost's extra flexibility nearly perfectly fits the *simple* stress relationship without overfitting, but the same flexibility overfits the *more complex* deflection relationship (compounding power laws + non-monotonic material effect from Day 39). RF's averaging is more conservative and better suited to the smaller, harder target.

## Model Selection

**Chosen: keep both, per target - XGBoost for Stress, Random Forest for Deflection.**

- Stress: XGBoost's near-perfect, non-overfit fit is a clear, objective win.
- Deflection: Random Forest remains the better choice, better test R²/RMSE, a smaller overfitting gap, more robust on the harder, sparser target.

Using a single algorithm for both targets isn't a requirement - the two targets have genuinely different statistical character (smooth vs. complex/sparse), and this two-model choice is justified by metrics, not convenience. Both remain simple, reproducible (`random_state=42`), and easy to explain.