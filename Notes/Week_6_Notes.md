
# Week 6 Notes

  

- After finishing processing of the data, we ran the prepared datasets through the ```RandomForestRegressor``` model
- **Why Random Forest suits both models:** Random Forest is a good fit here because both stress and deflection follow nonlinear, power-law relationships with Blade Length and Chord rather than straight lines, and RF doesn't assume linearity the way regression does, it splits the input space into regions and fits local averages, so it can approximate curved relationships without any manual transformation of the data. It also handles the mix of numeric (Blade Length, Chord, Load) and one-hot categorical (Material) features naturally, splitting on either type without special treatment. With only **81 simulations total**, RF's ensemble-of-trees averaging also guards against overfitting better than a single model would on such a small dataset, giving a reasonably **robust baseline** with almost no tuning. **The trade-off** - seen more severely in the deflection model,  is that RF predicts by averaging training examples within leaf regions, so it can't extrapolate and tends to shrink extreme predictions toward the mean, which is exactly the weakness that shows up at the high-deflection end of your data.

- n_estimators = 100, random_state = 42

  

## Stress Model

### Performance Metrics

| Metric | Value |
|---|---|
| MAE | 1.747 MPa |
| RMSE | 2.234 MPa |
| R² | 0.939 |

**What each metric means:**

-  **MAE (Mean Absolute Error):** the average size of a prediction error, in the same units as stress (MPa), treating over- and under-predictions equally. On average, a prediction is off by about 1.75 MPa.

-  **RMSE (Root Mean Squared Error):** similar to MAE but squares errors before averaging, so it penalises large errors more heavily. RMSE (2.234 MPa) being somewhat larger than MAE (1.747 MPa), showing that a handful of larger errors are pulling it up, rather than all 17 test errors being uniformly sized.

-  **R² (Coefficient of Determination):** the proportion of variance in stress that the model explains, relative to just predicting the mean every time. An R² of 0.939 means the model captures ~94% of the variation in stress across the test set, a strong baseline result.

  

```Relative error context: the test set stress values span roughly 1.06–29.13 MPa (a range of ~28 MPa). An MAE of 1.747 MPa is therefore about 6.2% of the full output range, which is a more meaningful way to judge "how good" the raw MPa number is than the absolute figure alone.```

  

### Interpretations

- The Actual vs Predicted plot shows the 17 test points clustered close to the diagonal "perfect prediction" line, with no wild scatter or systematic curvature away from it. Combined with R² = 0.939, the model has clearly learned the dominant structure in the stress data (chiefly the Blade Length and Chord relationships established in the correlation analysis) rather than just memorising noise.

- The largest errors are not randomly distributed, they cluster at the high-stress end of the test set. The mean error (bias) across all 17 test points is small (+0.16 MPa), and errors split roughly 12 overpredictions to 5 underpredictions, but the 5 underpredictions include the largest errors in the set. This is a mmoderately systematic pattern, not scattered noise: the model tends to be slightly conservative (overpredicts) on ordinary cases but underpredicts the most extreme ones.

-  **Which simulations have the largest prediction errors?**

| Simulation ID | Actual (MPa) | Predicted (MPa) | Error (MPa) |
|---|---|---|---|
| BL1.2_C150_CBF_P1500 | 29.13 | 24.51 | +4.62 |
| BL1.6_C200_FBG_P1000 | 19.27 | 15.71 | +3.56 |
| BL1.6_C200_FBG_P1500 | 28.91 | 25.41 | +3.50 |
| BL1.6_C200_CBF_P1000 | 19.27 | 15.86 | +3.41 |
| BL1.2_C150_FBG_P500 | 9.71 | 12.79 | −3.08 |

  

- A ~6% typical error is more than good enough to rank candidate designs or filter out clearly unsuitable ones. However, the errors are **not small enough for final design sign-off**: the ~4.6 MPa miss on the highest-stress case is exactly the kind of underprediction that would be dangerous to trust near a safety margin.

  

## Deformation Model

### Performance
-   **Performance:** MAE ~ 11.39 mm, RMSE ~ 20.17 mm, R² ~ 0.696, noticeably weaker than the stress model (R² = 0.939).
-   **Actual vs Predicted plot:** points hug the diagonal reasonably well at low deflection, but fall increasingly below the line at high deflection, the model systematically **underpredicts** the largest deflections.
-   **Residual plot:** shows a clear **funnel/fan pattern**, residuals stay small and tight for low predicted deflection, then spread out to large positive residuals (actual > predicted) as predicted deflection increases. Not random scatter, this is a genuine structural weakness.
-   **Worst case:** `BL1.6_C200_FBG_P1500`, actual 136.4 mm vs predicted 71.3 mm, an error of ~65 mm (~48% relative error). The next-worst cases are also high-deflection, fibreglass-heavy configurations.
-   **Bias direction:** mostly overpredicts small/mid deflections slightly, but badly **_underpredicts_** the handful of largest deflections, same "regression to the training mean" behaviour as the stress model, just far more severe here.

### Comparison with Stress Model
-   **Target range is much wider.** Deflection spans ~0.34–136 mm (~400×) vs stress's ~1–29 MPa (~28×), harder for 64 training points to cover densely at the extremes.
-   **Compounding power laws.** Deflection depends on Blade Length^4 and Chord^-3 simultaneously steeper and more compounded than stress's L^2/Chord^-2 - so the output surface is more curved and harder to approximate with limited data.
-   **Small dataset, sparse at extremes.** Only 64 training rows means very few examples near the highest-deflection corner of the design space (long blade, fibreglass, high load) for the trees to average over.

## Feature Analysis
###  Stress Model

| Feature | Importance |
|---|---|
| Root Chord | 0.381 |
| Blade Length | 0.345 |
| Applied Load | 0.251 |
| Material_Fiberglass | 0.008 |
| Material_Aluminium | 0.007 |
| Material_Carbon fiber | 0.007 |

###  Deflection Model

| Feature | Importance |
|---|---|
| Root Chord | 0.290 |
| Blade Length | 0.274 |
| Material_Fiberglass | 0.264 |
| Applied Load | 0.152 |
| Material_Carbon fiber | 0.012 |
| Material_Aluminium | 0.009 |

- ```Material_Fiberglass``` is neglogible for stress but jumps to **3rd most important feature** for deflection

## Comparison with EDA
| Variable | Correlation rank (Stress) | RF Importance rank (Stress) | Correlation rank (Deflection) | RF Importance rank (Deflection) |
|---|---|---|---|---|
| Blade Length | 1st (r=0.549) | 2nd | 1st (r=0.458) | 2nd |
| Chord | 2nd (r=-0.530) | 1st | 2nd (r=-0.356) | 1st |
| Load | 3rd (r=0.441) | 3rd | 3rd (r=0.226) | 4th |
| Material | 4th (r~0.000) | negligible | 4th (r=-0.066) | **3rd** (via Fiberglass) |

- Mostly similar with EDA for Blade Length, Chord Length and Load
- But two noticeable differences
1. **Blade Length and Chord swap first place.** Correlation ranked Blade Length slightly ahead of Chord for both outputs; Random Forest ranked Chord slightly ahead of Blade Length for both outputs. This is a minor reordering within an already-close pair, not a contradiction.
2. **Material's importance changes completely for deflection.** Correlation showed Material as essentially irrelevant (r~-0.066) for deflection. Random Forest's one-hot importance shows `Material_Fiberglass` as the 3rd most important feature, which is not a **minor** reordering.

  
**Why might this happen?**

- **The Blade Length/Chord swap** likely comes from the fact that correlation measures a single global linear trend across the whole dataset, while Random Forest's impurity-based importance reflects how much each variable's actual splits reduce prediction error, which can respond differently to how the *sampled levels* of a variable are spaced and how sharply the output curves respond within that specific range. Since both variables follow steep power laws over the same three sampled levels (0.8/1.2/1.6 m and 150/200/300 mm), it isn't surprising that their relative ranking is sensitive to the analysis method.

- **The Material discrepancy is the important one, and it is fully explained by Day 33's finding.** Correlation used *label encoding* (Aluminium=0, Fiberglass=1, Carbon fiber=2), which forces a false numeric ordering onto a variable that has no real order. Since deflection's true ordering is Fiberglass > Aluminium > Carbon fiber, non-monotonic with respect to that encoding, Pearson correlation was structurally blind to the effect. Random Forest, using **one-hot** columns, doesn't need Material to be ordered at all; it can split directly on "is this Fiberglass?" and immediately captures the real effect. This is the textbook example of **correlation measuring only linear relationships, while Random Forest captures nonlinear (and here, non-ordinal categorical) relationships** that correlation is mathematically incapable of seeing.

## So is it consistent with beam bending theory and mechanics?

- **Stress being geometry/load-dominated and material-independent is exactly what beam theory predicts.** Bending stress is σ = M*c/I,  a function of applied moment and cross-sectional geometry only. The elastic modulus E does not appear in this equation at all, so a well-trained model *should* find Material irrelevant for stress, and it does, with all three material columns sitting at under 1% importance.

- **Deflection depending strongly on Material is exactly what beam theory predicts, too.** Deflection is δ = (w*L^4)/(E*I), here E appears directly in the denominator, so a stiffer material produces less deflection at the same geometry and load. Fiberglass having a large importance score (rather than Aluminium or Carbon fiber) also makes sense: it's the material with the lowest stiffness in the set, producing the most extreme deflection values, so it's the material a tree can use to explain the most variance with a single split.

- **Chord and Blade Length dominating both outputs matches the power-law scaling discussed earlier** (Stress ∝ BladeLength^2/Chord^2, Deflection ∝ BladeLength^4/Chord^3) — geometry has the strongest and most compounding effect on both outputs, which is reflected in both models ranking these two variables at the top.
- **Load ranking below geometry, but still clearly relevant, is reasonable** - load enters linearly in both governing equations, while Blade Length and Chord enter as high powers, so geometry changes cause proportionally larger swings in the output across the sampled ranges, and the model reflects that difference in the achievable variance reduction.
