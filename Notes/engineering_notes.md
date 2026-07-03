**TURBINE GEOMETRY**

* MoM insights from syllabus and textbook
* Use it when you want to apply the physics
* Simulation will be done based on assumptions made and then it will be optimised based on Singapore weather patters
* Can I make one that is able to handle storms? Too many variables to take in
* Wind turbine is basically couple of airfoils rotating that generate lift
* Blade length - root to tip, chord length, width of blade section
* Blade angle changes from root to tip to optimise the angle of attack of the wind to reduce drag and hence increase lift
* turbines taper to reduce mass, tip loading, improve efficiency and reduce bending stress
* possible simplified model instead of distributed load can be w(x) = w (1-x/L)
* Power generated is proportional to the area swept by the blades
* Fillets are important to reduce stress concentration
* To reduce peak stress, smooth transitions are preferred rather than sharp ones, this is why fillets are important in the blade
* blade thickness changes from thick to thin, as deflection and bending stress are inversely proportional to thickness (moment of inertia). If an object is thin, it is hence structurally weaker and it is not able to handle stresses. It affects frequency too as it depend on mass and stiffness, which reduces resonance.
* Taper ratio is usually 0.25-0.35, 0.3 should be fine
* Formula used is interpolation for the different offset planes





**CAD NOTES**

* **Goal is stable parametric geometry**
* Parametric Design - Changing dimensions updates entire model automatically
* Important CAD features includes, Extrusion, Revolve, Fillet, Loft
* X for construction lines, D for dimension, keep saving your file
* Pattern creation makes it easier for designing instead of manually computing and duplicating, Use pattern creation under SOLID section
* S for sketch shortcuts
* Had trouble removing the kink in the leading edge of the airfoil cross-section **WHAT SHOULD I DO??**
* **Lofting is fun but can go really deep**
* CAD is easy to learn but so deep, tough to master id say
* Use model fillets 90% of the time, but use sketch fillets for lofting and sweeping profiles
* direction of threading is decided by WHERE you click on the profile, hex nut concept is very important
* The design is highly unstable, almost impossible to parameterise splines



**SIMULATION IDEAS**

* Static stress analysis
* blade deflection analysis
* fatigue analysis
* modal vibrational analysis
* aerodynamic loading estimation
* optimisation



**ROUGH BLADE SKETCH ( IMPROVEMENTS )**

* **No twist yet — As you noted, the cross-sections are all at the same angle. Real blades twist significantly (\~10–15° or more) from root to tip to account for the variation in relative wind angle at different radii. This is the most critical aerodynamic feature remaining.**
* **Root section — The root looks like it may transition quite abruptly to a circular or blunt form. Real blades have a cylindrical root for hub attachment — worth checking that transition is smooth.**
* **Chord distribution — From the top view, the taper appears fairly linear. A more optimised blade often has a slightly wider maximum chord located \~25–30% from the root (a "shoulder"), then tapers to the tip. Worth considering depending on your design goals.**
* All of the above has been implemented, there is still one issue remaining
* the top part of the blade is not uniform, possibly because all the sketches might be a bit different from each other
* will have to work on this by scaling each sketch so that the geometry is identical with no room for errors
* fixed it by just adding a spline on the top surface, connecting all the sketches
* **ONLY NEED TO PARAMETERISE: CHORD LENGTH, TWIST, LENGTH OF BLADE**
* **Now i have refined it to NACA 4412 sketch which is hand-made**
* **formula for taper is c(r)= Root\_chord − (Root\_chord\*0.7)×(r/blade\_span)**
* **twist angle can be derived from AI chatgpt or claude**





**BEAM SIMULATION IS DONE**

* the deflection readings came out properly
* for applying the force it was a bit complex, will have to read up on how to apply variations
* readings of the maximum stress didn't match calculations possibly because of the fillet structure, HAVE TO CLEAR THIS
* UI is very deep and will have to spend more time on it



**BLADE SIMULATION INITAL ROUGH**

* all the diagrams look good and according to intuition
* the equivalent stress occurs at the root and hub attachment junction or near the root
* Max stress is not exactly at root due to taper, twist and changing geometry of the blade
* mesh quality only near the trailing edge, the sharp edge, is a bit poor, but that is a design tradeoff
* mesh size - 3 e-3 metre







**CHANGING PARAMETERS**

* thicker blade - stiffer
* longer blade - much larger blade deflection
* more taper - lighter but weaker tip
* higher E - lower deflection
* lower density - lighter blade









