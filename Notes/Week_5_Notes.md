# Week 5 Notes
- Finished with all 81 simulations
- All the data looks consistent with expectations and assumptions
- Now moving towards thorough EDA of the updated dataset using Python's Pandas, Seaborn and Matplotlib libraries

## Initial Inspection
- Shape - (81, 8)
- Index(['ID', 'Blade_Length_m', 'Root_Chord_mm', 'Material', 'Applied_Load_Pa',
       'Status', 'Max_Deformation_m', 'Max_Equiv_Stress_Pa'],
      dtype='object') - **All features in dataset**
-  No missing values, no duplicate rows
- Material names are standardised 
- Units verified and data types too
- Data is internally consistent
- Correlation heatmap is also consistent with initial physics expectations
