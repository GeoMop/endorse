# Samples of Bayesian inversion of the stochastic parameters

Data authors: Tomáš Janda, Petr Kabele (FS ČVUT)

The DFN parameters of the 5 fracture populations are fitted to the geological mapping of the fracture outcrops.
0. The fracture size range is set constant for each population (a_min, a_max)
1. Orientation parameters (dip,strike,kappa) are fitted independently using a deterministic calibration. 
   Resulting best fit is in the file `fixed_params.csv`
2. The fracture size distribution parameters (P30 and alpha) are fitted using Bayesian inversion. Its samples are in the 
   files `P30_alpha_pop*.csv`.
   It seems that the fracture samples has been splitted into populations and then parameters of each population has been fitted independently.
   So we should draw from these population independently.
   
Some datails about inversion are part of the report [TZ 747-2024 Puklinova konektivita](https://drive.google.com/file/d/1NMVgSr2QKdnKSdWVMSY2MFIA47YVAfwP/view?usp=drive_link), 
in particular Section 6.2.1.
