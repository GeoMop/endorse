# Input data from other simulations and research

**config.yaml** 
Example main config file with commented fields.

**_*.yaml**
Configuration files included from the `confgi.yaml`

**large_model.msh** 
Result of a HG model of a catchment scale at one test locality for the repository

**large_model_local.msh** 
`large_model.msh` with nodes in local coordinate system. The local system origin is currently fixed. 

**conc_flux_UOS.csv** 
Prescribed  concentration flux (kg/rok/m2) at the interface bentonite - rock
for times from 10 years up to 1e6 years, max at 39 years 
half of max at about 100years, timestep 1year up to 1000 years, after just few times with 
negligable conc.
!! Seems to be very short peak, what is the model for this.
What will be the time of leak? We should also model random leak times and compare it 

The file is not used as the prescription of the conc. flux is instable. The Robin type condition with prescribed concentration is used instead.
However, the concentration is not realistic.


**conc_flux_UOS_kg_y.csv**
The same data but with 'comma' separator for simple import to libre office and with time in years instead of seconds.


**accepted_parameters.csv**
Parameters of the forward model for the accepted samples of the Bayes inversion. 
Rows are individual accepted samples, columns are parameters.
The first column 'N' provides number of steps the chain stays at the accepted parameter.


**tunnel_mesh_cut_healed.msh**
2D model of TSX experiment. Outer boundery square [-50,50]^2, 
tunnel is cutted out as an ellipse 1.8m vertical semiaxis, 2.2m horizontal semiaxis
EDZ region elipse: 8m vertical, 10m horizontal (its boundary is present, but the region is not resolved)    

**rectangle_2x5.msh**
Test 2d mesh , no particular reason. 
