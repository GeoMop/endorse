# Endorse - EDZ safety indicator simulations

The software implements specialized safety calculations for the excavation disturbed zone (EDZ)
of a deep repository of radioactive waste. It consists of two parts: 
1. determination of the rock parameters using the Bayesian inversion
2. stochastic prediction of the contamination transport and safety indicator evaluation

It essentially use the simulator [Flow123d](https://flow123d.github.io/) of processes in fractured rocks.

## Prerequisities

The software requires a working [Docker Desktop](https://www.docker.com/) 
installation or [SingularityCE](https://docs.sylabs.io/guides/latest/user-guide/quick_start.html) installation.
The first is better for local desktop usage, while the latter is usually the only option on HPC clusters. 
The use of clusters is recommended, as stochastic simulations are pretty computationally demanding. 
Currently, only the Linux installations are tested but should run 
with little effort on Windows due to containerization.


## Quick start

1. Download latest version of the sources as a ZIP package.
2. Extract to directory of your choice.
3. Setup the computational container with proper environment, using the `bin/endorse-setup` tool.
3. Create a work directory on a filesystem shared between computational nodes.
4. Prepare main configuration files.
5. Run Bayes inversion (`bin/endorse-bayes`) or stochastic transport (`endorse-mlmc`).

See [full documentation](doc/main.md) for the details.


of thefor the rock properties from the pore pressure measurements during 
the excavation. excavation damage zone (EDZ) properties and stochastic contaminant transport 
in order to provide stochastic prediction of EDZ safety indicators. 

The safety indicator is defined as the 95% quantile of the contaminant concentration on the repository model boundary over the whole simulation period. 
The contaminant is modeled without radioactive decay as a inert tracer. The multilevel Monte Carlo method is 
used to parform stochastic transport simulation in reasonable time. Random inputs to the transport model 
include: EDZ parameters, random hidden fractures, (random leakage times for containers), perturbations of the rock properties.

The EDZ properties are obtained from the Bayesian inversion, using data from pore pressure min-by experiment.
The Bayesian inversion provides posterior joined probability density for the EDZ properties (porosity, permability) as heterogenous fields.
That means the properties are described as correlated random variables. 


## Acknowledgement

Development of the advanced simulations and stochastic methdos for EDZ safety calculations and implementation of the software
was suported by:

![[TAČR logo]](logo_TACR_zakl.pdf) Technological agency of Czech republic
in the project no. TK02010118 of the funding programe  Theta.

### Authors

[** Technical university of Liberec **](www.tul.cz)

- **Jan Březina** coordination, stochastic transport
- **Jan Stebel** hydro-mechanical model in Flow123d
- **Pavel Exner** Bayes inversion for the EDZ
- **David Flanderka** Flow123d, optimizations, technicalities 
- **Martin Špetlík** [MLMC](https://pypi.org/project/mlmc/) library and homogenization
- **Radek Srb** containerization

[** Institute of Geonics **](https://www.ugn.cas.cz/?l=en&p=home)

- **Stanislav Sysala** plasticity model
- **Simona Bérešová, Michal Béreš** core Bayes inversion library [surrDAMH](https://github.com/dom0015/surrDAMH)
- **David Horák, Jakub Kružík** [PERMON](http://permon.vsb.cz/) library integration for fracture contacts in Flow123d


## Developers corner


### Repository structure:

- `doc` - software documentation and various reports from the Endorse project
- `experiments` - various numerical experiments and developments as part of the Endorse project
- `src` - main sources
- `tests` - various software tests, test data



### Development environment
In order to create the development environment run:

        setup.sh
        
As the Docker remote interpreter is supported only in PyCharm Proffesional, we have to debug most of the code just with
virtual environment and flow123d running in docker.
        
More complex tests should be run in the Docker image: [flow123d/geomop-gnu:2.0.0](https://hub.docker.com/repository/docker/flow123d/geomop-gnu)
In the PyCharm (need Professional edition) use the Docker plugin, and configure the Python interpreter by add interpreter / On Docker ...

