### Before running
- specify Docker image in file `endorse_fterm`
- call `endorse_fterm_sing` which creates SIF image, if it does not exist already
- open container using `endorse_fterm_sing`
and run `setup_python_enviroment.sh` to create Python venv with necessary packages


### Running simulation
- currently, running on Metacentrum cluster (TUL Charon) is supported
- the starting point is the execution script `run_all_metacentrum.sh`:
    ```./run_all_metacentrum.sh -n <N_CHAINS> -o <OUTPUT_DIR> -c -d```
    
    with parameters
    - `N_CHAINS` - number of Markov chains in Bayes
    - `OUTPUT_DIR` - workdir for all auxiliary, input and output files

- script `run_all_metacentrum.sh` opens container
  and runs the main script `run_all.py`
  and eventually runs the prepared PBS script with `qsub`
- script `run_all.py` does:
    - creates workdir, sets paths
    - calls `preprocess.py`, which
      - loads main config
      - reads and interpolates measurement data (`measure_data.py`)
      - creates mesh (`mesh_factory.py`)
      - sets some Bayes parameters to the template `config_mcmc_bayes.yaml`
    - copies input files to workdir
    - creates PBS script


### Obsolete/unused files
- `process.py`
- `run_all_local.sh`
- `run_collect_flow123d.sh`