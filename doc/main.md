## Endorse software

The package provides stochastic simulation tools for the characterization of the safety 
of an excavation damage zone of a deep geological repository of radioactive waste. 
The safety is described by the safety indicator – the maximum of a simulated concentration 
on the boundary of the computational domain. The Darcy flow and transport of the contaminant 
are calculated by the [Flow123d](https://flow123d.github.io/) simulator. The simulator uses 
the discrete fracture-matrix (DFM) approach that combines a network of (random) fractures and 
a 3D continuum. The safety indicator is considered a random variable dependent on the random 
fracture network and uncertainties in other parameters of the model. The stochastic properties 
(mean, variance, …) of the safety indicator are estimated using the Multilevel Monte Carlo (MLMC) 
method. The whole stochastic calculation is executed through the endorse-mlmc script described below.

The hydraulic conductivity and porosity on the excavation disturbed zone (EDZ)
are the key parameters affecting the safety indicator. These parameters are 
substantially increased in the vicinity of the tunnels of the repository compared
to the intact rock, partly due to damage attributed directly to the excavation 
method and partly due to deformations caused by changes in the stress field. 
However, the response to the stress changes is not immediate due to the 
presence of water. The continuous measurement of the pore pressure close to 
the excavated tunnel is used to determine the parameters of a poroelastic problem 
describing the relaxation of the EDZ. The Bayes inversion is used to obtain modified 
fields of hydraulic conductivity and porosity on the EDZ as the set of random samples. 
These can later be used as the input to the stochastic prediction of the safety indicator. 
The Bayes inversion is realized through the endorse-bayes script described below.

More details about the context, the used numerical models, and stochastic methods 
can be found in the methodology for characterization of the safety of EDZ. 
Currently only in Czech: “Posouzení vlivu zóny ovlivněné ražbou na bezpečnost 
hlubinného úložiště pomocí výpočetních metod”, link at the main repository page.
