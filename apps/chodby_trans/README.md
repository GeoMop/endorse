# "Chodby" project - transport model sensitivity analysis

Sensitivity analysis of transport model parameters using Sobol indices.
The transport model simulates radionuclide leakage from a waste disposal borehole
through the engineering barrier and surrounding fractured rock.
The geometry includes a single gallery with seven vertical waste
disposal positions and a random DFN.

## Structure

### [input_data](input_data)
The directory contains all the input data for sensitivity sampling.
- `fr_Bukov_bayes` - fracture families data from bayes inversion
- `_fr_Bukov_repo.yaml` - fracture families data
- `_fr_fixed.yaml` - single fracture family definition of the large fixed fracture
- `_geometry.yaml` - model geometry parameters
- `_ot_sensitivity.yaml` - sensitivity analysis settings 
- `transport_fullscale_tmpl.yaml` - template input file for Flow123d
- `trans_mesh_config.yaml` - main input file (mesh, time axis, metacentrum, ...)

### [input_variants](input_variants/README.md)
Main configurations for different cases researched in project Chodby. 

### [mesh](mesh)
Contain scripts that create mesh to the transport model.

### [dask_test](dask_test)
Testing od dask scheduler.

### [open_turns_sa_test](open_turns_sa_test)
Unit testing of Open Turns library (sensitivity analysis).

### [sa](sa)
[Deprecated] Auxiliary functions for SAlib library (sensitivity analysis).


## Install and environment
Use auxiliary bash script to create virtual python environment.
It installs necessary python modules and the repository module itself.
```
bash setup_env.sh
```
Later, use the virtual environment to run all the scripts.


## Running
The main script to run is `sensitivity_sampling.py`,
which can be run in several regimes:
- directly (PBS job submit or postprocess) 
- using `dask` scheduler locally
- using `dask` scheduler on Metacentrum

Several commands can be passed:

```>> python sensitivity_sampling.py <workdir> <command> <scheduler>```

including target working directory (where all the inputs will be copied and results created later),
one of the commands: `submit/read/continue/select/plots/meta/local` and `<scheduler>` if running under `dask`.

### Submit sensitivity sampling [Metacentrum]
- Call ```>> python sensitivity_sampling.py <workdir> submit```

    to create a workdir, copy input_data and submit PBS job for parallel sampling.
    Once the job starts, `dask` takes care of scheduling the samples on available workers
    across cpus and nodes.
    The samples output is collected into zarr storage inside the `<workdir>`.

- Call ```>> python sensitivity_sampling.py <workdir> continue```

    with `<workdir>` already containing some samples. Can be used when walltime was previously reached
    or when something went wrong during the sampling job. This collects all the samples that
    have not been computed yet and creates a new PBS job for sampling to continue.

### Postprocessing commands
- Call ```>> python sensitivity_sampling.py <workdir> read```

    to read the samples from zarr storage and print statistics of failed and valid samples

- Call ```>> python sensitivity_sampling.py <workdir> plots```

    to process the collected samples and run the actual sensitivity analysis.
    This computes the Sobol indices and makes all the plots to visualize results.

- Call ```>> python sensitivity_sampling.py <workdir> select <i_eval>```

    to run a selected sample with index `<i_eval>`.
    Use it particularly for running a selected sample with full output (Flow123d VTK, etc.).


### Running dask locally
Use auxiliary script in several steps to run the application:

- `bash dask_cluster_local.sh <workdir> start <N>`

    Starts the dask scheduler and `<N>` workers.

- `bash dask_cluster_local.sh <workdir> run <app_command>`

    Runs the `sensitivity_sampling.py` with `<app_command>`
    (e.g. `local`) on the created workers (it passed the scheduler).

- `bash dask_cluster_local.sh <workdir> stop`

    Stops the dask workers and scheduler in a clean way.


