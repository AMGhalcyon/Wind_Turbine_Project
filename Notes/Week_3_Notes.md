# Week 3 Notes

* Blade dimensions, NACA 4412 profile, 200mm root chord length with taper formula and twist, blade\_length - 1.2 m, hub attachment is 0.2m extended behind
* mesh size - 3 e-3 metres as found from mesh convergence
* trying to increase mesh quality near trailing edge
* applied pressure - 1000 Pa on top of blade structure
* mesh near trailing edge is very poor because of its sharpness, which is a drawback of the design
* formula for taper is c(r)= Root\_chord − (Root\_chord\*0.7)×(r/blade\_span)
* twist angle can be derived from AI chatgpt or claude( 14 - 
* **Deflection observed is similar to expected result, maximum being at the tip of the blade**
* **Max stress is near the root at the junction between the blade and the hub attachment**
## Materials Study 
Aluminium Alloy - Density = 2700 kg/m^3, Young's Modulus = 6.9 e+10 Pa, Poisson Ratio = 0.33
Fiberglass - Density = 1900 kg/m^3, Young's Modulus = 2.5 e+10 Pa, Poisson Ratio = 0.22
Carbon Fiber Composite - Density = 1600 kg/m^3, Young's Modulus = 1.2 e+11 Pa, Poisson Ratio = 0.28
* **Material Study Results**



|**Material**|**Deformation (m)**|**Equivalent Stress (Pa)**|
|-|-|-|
|Aluminium Alloy|0.010566|1.0813 e+7|
|Fiberglass|0.029204|1.0819 e+7|
|Carbon Fiber Composite|0.0060797|1.0816 e+7|

* **Carbon Fiber** is the stiffest material due to its high Young's Modulus. It exhibits smallest deformation while also being the lightest
**Aluminium Alloy** provides a balance between stiffness and weight, producing moderate deformation
**Fiberglass** has the lowest Young's Modulus, making it the least stiff. It is therefore expected to experience the largest deflection under the same loading conditions, although it is lighter than aluminium
All the materials produce similar equivalent stress readings, as it does not depend on stiffness
## Load Study 
- Applied 500 Pa, 1000 Pa, and 1500 Pa to check relationship between load and deformation and with stress too



|**Load (Pa)**|**Deformation (m)**|**Equivalent Stress (Pa)**|
|-|-|-|
|500|0.0052829|5.4065 e+6|
|1000|0.010566|1.0813 e+7|
|1500|0.015849|1.622 e+7|



* Loading has a linear relationship with deformation and stress, doubling load will in fact double the other two quantities too
* Stress location remains unchanged, and deformation plots remain similar too
* This workflow can be used as a repeatable process for generating simulation data for future machine learning model development
## GEOMETRY STUDY (VERY IMPORTANT)
* thicker blade - stiffer
* longer blade - much larger blade deflection
* more taper - lighter but weaker tip
* higher E - lower deflection
* lower density - lighter blade
* Changing hub attachment radius barely changes readings, going to maintain 100mm diameter and 1000 Pa load
* **CHORD LENGTH VARIATION - keeping 1.2 m blade length constant**

|**Chord Length (mm)**|**Deformation (m)**|**Equivalent Stress (Pa)**|
|-|-|-|
|150|0.024251|1.9419 e+7|
|200|0.010566|1.0813 e+7|
|300|0.0032622|4.8257 e+6|



* The relationship is clearly nonlinear — each step up in root chord gives a larger percentage improvement than the last, consistent with beam theory where bending stiffness scales roughly with the cube of cross-sectional size and bending stress scales roughly with the square. So increasing root chord is a disproportionately effective lever for reducing both deflection and stress, not a linear trade-off.
## BLADE LENGTH STUDY 
- KEEPING 200 mm Root Chord Length Constant

|**Blade Length (m)**|**Deformation (m)**|**Equivalent Stress (Pa)**|
|-|-|-|
|1.2|0.010566|1.0813 e+7|
|0.8|0.0021496|4.7955 e+6|
|1.6|0.032908|1.9266 e+7|



* Both deformation and stress increase sharply, and accelerate, as blade span increases - the opposite trend from the chord study, and far more aggressive. This is exactly what cantilever beam theory predicts: : for a fixed tip load, deflection scales with span cubed (δ ∝ L^4/EI) and bending stress scales with span (σ ∝ M/Z = (load×L)/Z), so a longer blade is a much worse lever arm in both ways at once.
* 1000 Pa pressure applied surface pressure that produces an effectively distributed line load w(r) = 1000 × c(r) N/m along the span, and note that despite the non-uniform (tapered) nature of this load, your simulation results still empirically matched the L⁴ (deflection) and L² (stress) scaling laws associated with the simpler uniform distributed-load cantilever model
## LIMITATIONS
- Static loading only
- Simplified wind loading
- No aerodynamic analysis
- Isotropic Material assumption
- No fatigue analysis



## NEXT STEPS 
- Generate 300-500 simulation cases
- Build structured dataset (CSV)
- Begin automation where possible 
- Prepare data for ML

