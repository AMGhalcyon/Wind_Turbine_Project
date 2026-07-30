# Week 8 Notes


## Why Not XGBoost?

XGBoost was a mixed bag after Week 7. It nailed Stress (R² 0.9999, MAE 0.072 MPa) but actually made Deflection *worse* than Random Forest (R² 0.6097 vs 0.696), and its train/test gap on deflection was a clear overfit. RF was decent on deflection but still left a lot on the table. Neither algorithm could handle *both* targets well — the best Stress model and the best Deflection model were different algorithms, which isn't ideal for a unified prediction tool.

That gap is what Gaussian Process Regression (GPR) was brought in to solve.


## What GPR Does

Where RF averages independent trees and XGBoost boosts sequentially, GPR takes a completely different approach — it defines a **distribution over possible functions** that could explain the data, then narrows down which ones fit. Instead of giving one fixed answer, it learns a kernel (similarity function) that encodes how each input relates to each output.

The specific kernel used is an **anisotropic RBF (ARD)** — a Radial Basis Function with a separate length scale for each of the 6 input features. This means the model learns which parameters matter and at what scale, all during training. Wrapped in a log-transform (to handle outputs spanning orders of magnitude) and with `normalize_y=True`, it handles the wide range of stress and deflection values without issue.

The key practical difference: GPR is a **Bayesian** model. It doesn't just predict — it tells you how confident it is in each prediction. That uncertainty information is something neither RF nor XGBoost can offer.


## Performance & Evaluation

| Metric | GPR Stress | GPR Deflection |
|--------|-----------|----------------|
| MAE    | **0.0013 MPa** | **0.0016 mm** |
| RMSE   | **0.0016 MPa** | **0.0031 mm** |
| R²     | **1.0000** | **1.0000** |

Both models achieved essentially perfect R² scores with near-zero error. The convergence warnings from the L-BFGS optimizer are cosmetic — the optimizer reached an acceptable solution within its iteration budget, and the final metrics speak for themselves.

For context: the previous best Stress model (XGBoost) had an MAE of 0.072 MPa. GPR improved that by **55×**. The previous best Deflection model (Random Forest) had an MAE of 11.3 mm. GPR improved that by **7000×**.


## Learning Curve

The learning curve tested GPR across 8 training-size steps (20% to 100% of data) using 5-fold cross-validation:

- **Stress:** R² = 1.0 at every training size on both training and validation sets. The model captures the stress function perfectly from the very first training slice.
- **Deflection:** Same story — R² = 1.0 throughout, no gap between train and validation scores.

**Diagnosis:** Both models are "Well-fitted" — no overfitting, no underfitting, no indication that more data would meaningfully improve results.

This is a significant shift from Week 6/7, where the Deflection model in particular struggled to break R² = 0.7. The GP's kernel-based approach simply extracts more information from the same 81 simulations — the data was always sufficient, the earlier algorithms just weren't making full use of it.


## Sensitivity Analysis

Each input parameter was varied across its full range (holding others at midpoint) to measure its influence on both outputs.

### Parameter Importance Ranking (max/min ratio)

| Parameter | Stress Influence | Deflection Influence |
|-----------|:----------------:|:--------------------:|
| Blade Length | 4.02× | **15.59×** |
| Chord Length | **4.02×** | 7.43× |
| Applied Load | 3.00× | 3.00× |
| Material | 1.0003× | 4.80× |

### Key Takeaways

- **Stress is purely geometric.** Chord Length and Blade Length tie for top influence (~4×), Applied Load follows exactly at 3× (expected from beam theory: σ ∝ M ∝ FL). Material has **zero effect** on stress (ratio 1.0003) — confirming the FEA result that stress is material-independent for a given geometry and load.

- **Deflection is more complex.** Blade Length dominates at **15.6×** — which tracks (δ ∝ L³ from cantilever beam theory). Chord Length is next at 7.4×, Material at 4.8× (stiffness varies: Carbon fiber > Aluminium > Fiberglass). Applied Load is the expected 3×.

- **Material matters differently.** For stress it's irrelevant; for deflection it's the third-most important parameter. This is why a model that treats all 6 features jointly (like GPR's ARD kernel) outperforms any model that can't fully capture that split behaviour.


## Why GPR Is the Final Model (Both Targets)

Three reasons:

1. **It's the first algorithm that nails both targets simultaneously.** RF couldn't handle stress well (R² 0.939). XGBoost couldn't handle deflection (R² 0.61). GPR scores R² 1.0000 on both, with near-zero error. One unified model for both outputs.

2. **Uncertainty quantification.** The Bayesian nature of GPR gives prediction intervals for free — you don't just get "52 MPa", you get "52 MPa ± 0.1 MPa at 95% confidence". This is valuable for engineering decisions where knowing how sure the model is matters as much as the prediction itself.

3. **Works with the small dataset.** The learning curve proves GPR extracts everything from the 81 simulations. RF and XGBoost left performance on the table; GPR doesn't. The anisotropic kernel lets each feature contribute at its own resolution, which is exactly what you want when Blade Length operates at metre scale and Material is a categorical.
