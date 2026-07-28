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



## Cross Validation

### Why Cross-Validation?

With only 81 simulations, an 80/20 split leaves just 17 rows in the test set. That's a small enough sample that a few unusual rows, either unusually easy or unusually hard to predict, can swing MAE, RMSE, and R² quite a bit. Previously whole RF vs XGBoost comparison rested on one such split, so there was no way to know if those numbers were real or just a lucky (or unlucky) draw.

5-fold CV fixes this by rotating which 1/5 of the data is held out, training the model 5 times total. Every row gets tested on exactly once. Instead of one score, we get 5, and the spread of those 5 tells us how much to trust the average.

## Results

Same final models as previously taken (RF: `n_estimators=100`; XGBoost: `n_estimators=100, max_depth=4, learning_rate=0.1`), run on the full dataset with `KFold(n_splits=5, shuffle=True, random_state=42)`. Scaling was refit inside each fold to avoid leakage.

| Model | Target | Mean MAE | Mean RMSE | Mean R² | Std Dev (R²) |
|---|---|---|---|---|---|
| Random Forest | Stress | 1.80 MPa | 2.48 MPa | 0.941 | 0.027 |
| XGBoost | Stress | 0.08 MPa | 0.13 MPa | 0.9997 | 0.0005 |
| Random Forest | Deflection | 14.00 mm | 24.69 mm | 0.375 | 0.466 |
| XGBoost | Deflection | 11.89 mm | 21.99 mm | 0.644 | 0.080 |

The number that matters most here isn't the mean, it's Random Forest's Deflection std dev of 0.466. Per-fold, its R² ranges from **-0.55 to 0.69**. One fold was worse than just guessing the average. XGBoost's Deflection R² only ranged 0.55–0.77, nowhere near that swing.

**Answers:**
- **Most stable:** XGBoost, on both targets, no contest.
- **Least fold-to-fold variation:** XGBoost — Random Forest's Deflection score literally goes negative on one split.
- **Overfitting:** Random Forest's previous grid search already showed a persistent train/test gap on Deflection (~0.27). That same fragility shows up here as fold instability. XGBoost's shallower trees + boosting generalize more evenly.

### Interpretation

**More reliable model:** XGBoost. Random Forest's Day 43 Deflection score of 0.696 R² looks solid on its own, but CV shows that number could easily have been 0.5 or even negative with a different split. That's not a model you can stand behind with 81 data points.

**Generalizes better:** XGBoost, especially on Deflection, where the *consistency* gap is the real finding, not just the accuracy gap.

**Is Deflection harder than Stress?** Yes, clearly, for both models. Stress scales in a near-linear, almost textbook way with load and geometry, so it's easy to nail. Deflection depends on a messier mix of stiffness, geometry, and material — harder to model, and it shows in both models' lower and noisier R².

**Dataset size matters:** With ~16 rows per test fold, one odd geometry can tank a fold's score. That's exactly what happened to Random Forest. It's not a coding bug, it's the small dataset doing what small datasets do, and it's why we ran this test in the first place.

## Model Selection Update

Previous findings hinted XGBoost was ahead. This confirms it wasn't a fluke — across 5 independent splits, XGBoost beats Random Forest on both accuracy and (more importantly) stability.

**Final call: XGBoost**, for both Stress and Deflection, going forward. Random Forest stays useful as a baseline and for its feature importances, but it's not the model to trust on a dataset this size.


## Learning Curves
Using the XGBoost model, I plotted training score vs. validation score as training set size grows from ~12 to 64 rows.

**Stress:** Training R² sits at ~1.0 the whole way. Validation R² starts poor with few samples (even negative at n=12) but climbs steadily and converges to 0.999 by n=64. The gap between the two curves shrinks to almost nothing. (converges)

**Deflection:** Training R² stays pinned near 0.998 throughout. Validation R² is noisy and lands at only 0.579 at full training size, a persistent, large gap that never closes.

### Model Diagnosis

| Model | Training Score | Validation Score | Diagnosis |
|---|---|---|---|
| Stress | 1.000 | 0.999 | Well-fitted |
| Deflection | 0.998 | 0.579 | Overfitting |

The Stress curves converging tells the story on their own, the model isn't just memorizing, it's actually learning the relationship.

Deflection is the opposite story: near-perfect training score paired with a validation score that's both much lower and still bouncing around at n=64 is the classic overfitting sign, the model fits the training rows well but that fit isn't transferring reliably to new ones. XGBoost generalizes better than a plain average here, but it's still the weaker of the two models, consistent with previous finding that Deflection is a harder target to capture.

## Dataset Discussion

81 simulations is enough to prove the concept works, Stress prediction is essentially solved at this sample size, and even Deflection beats a naive baseline. But it's not enough to fully close the gap: the validation curve is still climbing and still noisy at the largest training size tested, which is a direct signal (not a guess) that more data would likely help.

Deflection is harder because it depends on a more complex combination of geometry and material stiffness than Stress, which is closer to a straightforward function of load and cross-section. With few examples spanning that combined space, the model has less to learn the pattern from.

Generating more ANSYS simulations was outside the scope of this project — each one takes real setup and compute time, and 81 was what fit the project timeline. This is a genuine constraint, not a shortcut.

### Possible Future Works

- Run more ANSYS simulations, prioritizing the deflection-sensitive region of the design space
- Add more airfoil profiles beyond what's currently covered
- Test additional blade materials
- Widen the range of blade lengths and chord lengths tested
- Try additional algorithms (e.g., Gradient Boosting variants, Gaussian Process Regression) that may handle small, noisy targets like Deflection better


## Sensitivity Analysis


Final XGBoost model (trained on all 81 rows) was swept across each design parameter, holding the others fixed at Blade Length = 1.2 m, Chord = 150 mm, Material = Carbon Fiber, Load = 1000 Pa.

**Blade Length (0.8 -> 1.6 m):** Stress 8.6 -> 34.5 MPa (4.0x). Deflection 2.0 -> 46.2 mm (22.9x). Deflection blows up far faster than stress — matches beam theory, where deflection scales with L⁴ but stress only with L².

**Chord Length (150 -> 300 mm):** Stress 19.4 -> 4.9 MPa (drops 4.0x). Deflection 15.1 -> 0.78 mm (drops 19.4x). A wider chord stiffens the blade dramatically, this tracks the chord³ term in the second moment of area.

**Applied Load (500 -> 1500 Pa):** Stress 9.7 -> 29.1 MPa (3.0x, essentially linear with load, as expected for linear elastic behavior). Deflection 6.6 -> 24.2 mm (3.6x, close to linear too).

**Material (bonus check, supporting the ranking below):** Stress is basically identical across Aluminium, Fiberglass, and Carbon Fiber (~19.4 MPa each) since stress here mostly depends on geometry and load, not stiffness. Deflection is a different story, Fiberglass deflects 63.5 mm vs Carbon Fiber's 15.1 mm (4.2x), because Fiberglass is far less stiff.



## Parameter Ranking

Ranked by max/min ratio across each sweep:

| Rank | Variable | Stress Ratio | Deflection Ratio | Influence |
|---|---|---|---|---|
| 1 | Blade Length | 4.0x | 22.9x | Very High |
| 2 | Chord Length | 4.0x | 19.4x | Very High |
| 3 | Material | ~1.0x | 4.2x | Moderate |
| 4 | Applied Load | 3.0x | 3.6x | Moderate |

Blade Length and Chord Length dominate, both are geometric parameters that compound through the beam-stiffness equations. Material and Load have real but comparatively smaller effects, and load's effect is close to linear rather than compounding.

## Structural Recommendations

**To reduce deflection:** shorten the blade or widen the chord, both are far more effective than switching materials or reducing load. Chord length is the standout lever here since widening it cuts deflection by nearly 20x across the tested range.

**To reduce stress:** blade length and chord length again dominate, roughly symmetrically (~4x each); load has a smaller, linear effect; material barely matters for stress at all.

**Trade-off:** a shorter blade and wider chord both reduce stress and deflection, but a wider chord adds mass, drag, and material cost, and a shorter blade cuts into swept area (and therefore energy capture), the ML model can't quantify that trade-off itself, since it wasn't trained on cost or aerodynamic data, but it makes clear that geometry decisions carry far more structural weight than material choice.

## Code Refactoring 
Benefits of modular code
- **One source of truth.** `Max_Equiv_Stress_Pa` cleaning and the
  `Material` one-hot encoding used to be re-typed in every notebook, now
  it's one function (`data_processing.load_dataset`,
  `data_processing.encode_materials`), so a fix only has to happen once.
- **Consistent pipelines.** `training.build_pipeline` is the same
  ColumnTransformer + model pipeline used in the RF/XGBoost notebooks, the
  cross-validation notebook, and the learning-curve notebook. No more risk
  of one notebook scaling before the split and another scaling inside a
  Pipeline.
- **Shorter notebooks.** Notebooks now read as a narrative (load -> prepare ->
  train -> evaluate -> plot) 
- **Reusable for the prediction tool.** Later on the prediction tool
  (`predict.py`) reuses `data_processing`, `training`, and `prediction`
  directly — no reimplementation needed.


## Predict Tool for Users

Built a command-line tool that lets anyone input blade parameters and get instant stress and deflection predictions. The tool loads the trained XGBoost models, validates inputs (catches negatives, invalid materials), warns when values fall outside the training range, and displays results in a clean colored interface. Predictions are accurate to within 0.02 MPa of the actual simulation data. The entire workflow is reproducible—run `train_and_save_models.py` once to generate the models, then `predict.py` anytime to make predictions. This addresses the earlier gap of having models only in notebooks without a practical way for others to use them.







