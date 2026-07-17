
# Week 5 Notes

  

- Finished with all 81 simulations

  

- All the data looks consistent with expectations and assumptions

  

- Now moving towards thorough EDA of the updated dataset using Python's Pandas, Seaborn and Matplotlib libraries

  

  

## Initial Inspection

  

- Shape - (81, 8)

  

- Index(['ID', 'Blade_Length_m', 'Root_Chord_mm', 'Material', 'Applied_Load_Pa',

  

'Status', 'Max_Deformation_m', 'Max_Equiv_Stress_Pa'],

  

dtype='object') - **All features in dataset**

  

- No missing values, no duplicate rows

  

- Material names are standardised

  

- Units verified and data types too

  

- Data is internally consistent

  

- Correlation heatmap is also consistent with initial physics expectations

  

  

## EDA

  

- Most of the EDA was already done while validating the simulation process

  

### Observations are as follows

  

-  **Deformation vs Load** - perfectly linear for every material/geometry combination. Confirms linear-elastic FEA behaviour, no outliers

  

-  **Same applied for Stress vs Load**

  

-  **Materials Lines overlap** on the Stress plots, as it depends only on geometry and load, not stiffness (E) - this is expected from beam thoery

  

-  **Material Lines are clearly separated on the Deformation plots** - Fiberglass deflects the most, followed by Aluminium Alloy followed by Carbon Fiber Composite, in a consistent ~ 2.8:1.7:1 ratio across every geometry. This tracks 1/E almost exactly

  

-  **Deformation vs Blade Length** is steeply non-linear - fitted exponent ~ 3.96-3.97 (L^4 scaling), doubling the span increasing deflection almost 16x times

  

- For **Stress vs Blade Length**, doubling the span almost increases stress 4x

  

-  **Increasing Chord Length** can suppress both effectively. At 300mm, deflection curves are nearly flat across blade lengths, whereas in the case for 150mm, it rises steeply

  

-  **Bar charts (Fixed 1000 Pa)** - for a given material, deflection drops sharply as chord goes 150 -> 200 -> 300mm. The same drop is seen in Stress but milder, since it is chord-dominated and material independent

  

-  **Blade Length** is the single most sensitive parameter in the whole matrix, especially deflection

  

-  **Root chord** is the main lever for controlling deflection once span grows, changing the material cannot offset the massive trade-off by the blade length like the way increasing the chord can

  

## Correlation Analysis

| | Blade Length | Root Chord | Material (enc) | Applied Load | Stress | Deformation |
|------------------|:---:|:---:|:---:|:---:|:---:|:---:|
| Blade Length | 1.000 | 0.000 | 0.000 | 0.000 | 0.549 | 0.458 |
| Root Chord | 0.000 | 1.000 | 0.000 | 0.000 | -0.530 | -0.356 |
| Material (enc) | 0.000 | 0.000 | 1.000 | 0.000 | 0.000 | -0.066 |
| Applied Load | 0.000 | 0.000 | 0.000 | 1.000 | 0.441 | 0.226 |
| Stress | 0.549 | -0.530 | 0.000 | 0.441 | 1.000 | 0.753 |
| Deformation | 0.458 | -0.356 | -0.066 | 0.226 | 0.753 | 1.000 |

  

```(Material encoded as Aluminium = 0, Fiberglass = 1, Carbon fiber = 2, for this analysis only.)```

  

-  **Heatmap** can be found in the jupyter notebook

  

### Key Observations

-  **All four input variables are mutually uncorrelated ( ~ 0.00)** - this is the expected outcome. It means any importance ranking drawn from this matrix reflects each variable's own effect, not a side-effect of correlated inputs

- **Blade Length and Root Chord Length are the two strongest linear drivers of both outputs** - applied load is a mdoerate driver. Material shows almost no linear correlation with either output

-  **Material's near zero correlation is misleading**, it is the effect of the label encoding, not evidence that material doesn't matter

  

### Feature Importance

**Maximum Stress:**

  

1. Blade Length (0.549)

  

2. Chord Length (0.530)

  

3. Applied Load (0.441)

  

4. Material (0.000)

  

**Maximum Deflection:**

  

1. Blade Length (0.458)

  

2. Chord Length (0.356)

  

3. Applied Load (0.226)

  

4. Material (0.066)

  
  

- Deformation scales roughly as (Blade Length)⁴ and (Chord)⁻³ ( for chord, the simulation readings may be slightly off, ~ -2.89, probably because of the hub junction attachment. This trade-off is negligible though)

- Stress scales roughly as (Blade Length)² and (Chord)⁻²

- But due to Pearson's method of linear association, the correlation coefficients often underestimate the strength of these relationships. This is a limitation of the method, not the physics.

- This effects the importance for the 'Material' feature as actual deformation ordering is Fiberglass > Aluminium > Carbon fiber — i.e., the middle-coded material (Fiberglass) produces the _largest_ deformation, not a monotonic increase or decrease with the encoding. Pearson correlation can only detect monotonic linear trends, so a non-monotonic categorical effect like this is invisible to it regardless of how strong it is. This is precisely why **one-hot encoding** is used for the machine learning stage

  

## Dataset Preparation for Machine Learning

### Machine Learning Readiness

-  **Dataset quality is high** - The full DOE means the four inputs are orthogonal, so the model won't have to disentangle confounded effects and the outputs follow cleanly according to the power-law relationships from physics.

-  **Relationship strength is favourable once nonlinearity is accounted for** - The scatterplots show strong, smooth power laws - exactly the kind of structure that regression models can learn efficiently, provided the inputs are scaled properly

-  **Dataset size is the main limiting factor** - 81 simulations doesn't leave much room for train-test split or a high capacity model without overfitting.

-  **Material encoding must change**- one hot encoding must be used to get the proper effect of material dependence

### Processing for ML model integration
**Features used (X):**

-   Blade Length
-   Root Chord Length
-   Material (categorical: Aluminum, Fiberglass, Carbon Fiber)
-   Applied Load

**Target variables (y) — two separate models:**

-   Model 1: Maximum Equivalent Stress
-   Model 2: Maximum Deflection

- **Encoding method:** One-hot encoding on `Material`, producing three binary columns (`Material_Aluminum`, `Material_Fiberglass`, `Material_Carbon Fiber`) instead of an arbitrary integer label.

- **Scaling method:** `StandardScaler` (zero mean, unit variance), applied only to the continuous numeric features — Blade Length, Root Chord, Applied Load. The one-hot material columns are left unscaled.

- **Train-test split ratio:** 80% train / 20% test, `random_state=42` for reproducibility. The split is performed independently for the stress dataset and the deflection dataset (same X, different y, so the row split can differ between the two).

``` Now the dataset is ready for ML models ```