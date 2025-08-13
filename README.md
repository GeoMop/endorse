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

1. Download the latest version of the sources as a ZIP package.
2. Extract to the directory of your choice.
3. Set up the computational container with the proper environment using the `bin/endorse-setup` tool.
3. Create a working directory on a filesystem shared between computational nodes.
4. Prepare main configuration files.
5. Run Bayes inversion (`bin/endorse-bayes`) or stochastic transport (`bin/endorse-mlmc`).

See [full documentation](doc/main.md) for the details.


## DVC usage
[DVC](dvc.org) is used to separate version control of large datasets from processing codes and configurations stored on GitHub.
For endorse repository the large datasets are stored on Google drive under the shared drive DZ04_Chodby.

### Setup 

1. Use 'bin/dvc_install.sh' for pip install into a Python environment 
   or [DVC install](https://dvc.org/doc/install) for other options like install it system-wide.
   
2. Download DVC endorse secret [config script](https://drive.google.com/file/d/1Dag4N3KYz5q9rkLURayXHjUV0yN-zYYH/view?usp=drive_link),
   place it to endorse root under original name (NEVER COMMIT THIS FILE).

3. Execute the script like:

        ```
        bash dvc_secret_config.sh
        ```

4. Pull the large files:
        ```
        dvc pull
        ```
   The browser should open to ask you for the login to your Google account (the TUL one ussually).
   
   
See [large datasets modification doc](https://dvc.org/doc/user-guide/data-management/modifying-large-datasets) for further work.

### Adding remote (initialization)

1. Initialize `.dvc` folder. From the repository root run:

        ```
        dvc init
        ``` 

2. Add google drive remote [DZ04_Chodby/Podklady/endorse_large_files](https://drive.google.com/drive/u/1/folders/109cr1pZ8GV5s8yXKgVzl8NPQ8j537E4T)

        ```
        dvc remote add -d gdrive gdrive://109cr1pZ8GV5s8yXKgVzl8NPQ8j537E4T

        ```

The hash comes form the link.





## Acknowledgement


| <img src="./doc/logo_TACR_zakl.png" alt="TACR logo" height="80px"> |Development of the Endorse software was supported by <br> Technological agency of Czech republic <br>in the project no. TK02010118 of the funding programme Theta.|
|:---:|:---|
### Authors

**[Technical university of Liberec](www.tul.cz)**

- **Jan Březina** coordination, stochastic transport
- **Jan Stebel** hydro-mechanical model in Flow123d
- **Pavel Exner** Bayes inversion for the EDZ
- **Martin Špetlík** [MLMC](https://pypi.org/project/mlmc/) library and homogenization

**[Institute of Geonics](https://www.ugn.cas.cz/?l=en&p=home)**

- **Stanislav Sysala** plasticity model
- **Simona Bérešová** core Bayes inversion library [surrDAMH](https://github.com/dom0015/surrDAMH)
- **David Horák, Jakub Kružík** [PERMON](http://permon.vsb.cz/) library integration for fracture contacts in Flow123d

### Coauthors
- **David Flanderka** Flow123d, optimizations, technicalities 
- **Radek Srb** containerization
- **Michal Béreš** consultation, tests

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

