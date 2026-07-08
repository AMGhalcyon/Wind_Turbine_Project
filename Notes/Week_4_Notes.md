# DATASET BUILDING ( WEEK 4 )

## DESIGN VARIABLES


|**Variable**|**Levels**|
|-|-|
|Blade Length|0.8, 1.2, 1.6 m|
|Chord Length|150, 200, 300 mm |
|Materials|Aluminium Alloy, Fiberglass, Carbon Fiber Composite|
|Load (Low, Medium, High)|500, 1000, 1500 Pa|




**Total Simulations - 3 \* 3 \* 3 \* 3 = 81 simulations**



* **Now I have to create a simulation matrix to systematically conduct the simulations in an order**
* Within ANSYS project file, I will create 2 additional modules of ANSYS Mechanical so that I have 3 in total for the 3 different chord lengths. All the other variations I can add within the module itself then.

## Running Batches of Simulations

- Now for the next few days, I will be running 9-10 simulations each day with different combinations and entering this data in the csv file
- This is a MAJOR step in integrating ML into this project
- Conducted EDA on whatever data I have updated in the csv file. My python code runs on that data and generated graphs accordingly. SO I can keep updating my data as it goes and keep verifying along with it.

- After running a few batches of simulations it is clear that material choice significantly affects deformation but not stress. Carbon fiber deofrms the least, Fiberglass the most, while all three materials produce nearly identical von-Mises stress under the same load and geometry.
- This is because stress is governed by applied load and geometry, not material stiffness. Increasing chord_length reduced deformation substantially across all materials. 
- Key design implication - material selection controls deformation, geometry controls stress
- When the default mesh I am using is too fine and is too costly for the computer to calculate, I update it to 5 e-3 m for each element size

## After Running 54/81 simulations

- All the data correctly lines up with expected output and previous mathematical findings
- Blade length has the greatest influence on deflection, whereas increasing chord length singificantly reduces peak stress
- Material stiffness affects deformation but not stress

## Challenges

- Long simulation times
- Plugging in the different variations is time consuming 


## Week 5 Plan

- Finish remaining simulations, finalise dataset
- then import into python and analyse 
- after conducting thorough analysis, prepare the data for ML