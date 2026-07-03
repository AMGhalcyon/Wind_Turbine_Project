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