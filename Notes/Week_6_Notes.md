# Week 6 Notes

- After finishing processing of the data, we ran the prepared datasets through the ```RandomForestRegressor``` model
- n_estimators = 100, random_state = 42

## Stress Model 
### Performance Metrics
 
| Metric | Value |
|---|---|
| MAE | 1.747 MPa |
| RMSE | 2.234 MPa |
| R² | 0.939 |
 
**What each metric means:**
- **MAE (Mean Absolute Error):** the average size of a prediction error, in the same units as stress (MPa), treating over- and under-predictions equally. On average, a prediction is off by about 1.75 MPa.
- **RMSE (Root Mean Squared Error):** similar to MAE but squares errors before averaging, so it penalises large errors more heavily. RMSE (2.234 MPa) being somewhat larger than MAE (1.747 MPa), showing that a handful of larger errors are pulling it up, rather than all 17 test errors being uniformly sized.
- **R² (Coefficient of Determination):** the proportion of variance in stress that the model explains, relative to just predicting the mean every time. An R² of 0.939 means the model captures ~94% of the variation in stress across the test set — a strong baseline result.

```**Relative error context:** the test set stress values span roughly 1.06–29.13 MPa (a range of ~28 MPa). An MAE of 1.747 MPa is therefore about **6.2% of the full output range**, which is a more meaningful way to judge "how good" the raw MPa number is than the absolute figure alone.```

### Interpretations
- The Actual vs Predicted plot shows the 17 test points clustered close to the diagonal "perfect prediction" line, with no wild scatter or systematic curvature away from it. Combined with R² = 0.939, the model has clearly learned the dominant structure in the stress data (chiefly the Blade Length and Chord relationships established in the correlation analysis) rather than just memorising noise.
- The largest errors are not randomly distributed — they cluster at the high-stress end of the test set. The mean error (bias) across all 17 test points is small (+0.16 MPa), and errors split roughly 12 overpredictions to 5 underpredictions — but the 5 underpredictions include the largest errors in the set. This is a mmoderately systematic pattern, not scattered noise: the model tends to be slightly conservative (overpredicts) on ordinary cases but underpredicts the most extreme ones.
- **Which simulations have the largest prediction errors?**
 
| Simulation ID | Actual (MPa) | Predicted (MPa) | Error (MPa) |
|---|---|---|---|
| BL1.2_C150_CBF_P1500 | 29.13 | 24.51 | +4.62 |
| BL1.6_C200_FBG_P1000 | 19.27 | 15.71 | +3.56 |
| BL1.6_C200_FBG_P1500 | 28.91 | 25.41 | +3.50 |
| BL1.6_C200_CBF_P1000 | 19.27 | 15.86 | +3.41 |
| BL1.2_C150_FBG_P500  | 9.71  | 12.79 | −3.08 |

-  A ~6% typical error is more than good enough to rank candidate designs or filter out clearly unsuitable ones. However, the errors are **not small enough for final design sign-off**: the ~4.6 MPa miss on the highest-stress case is exactly the kind of underprediction that would be dangerous to trust near a safety margin.

## Deformation Model
