`tests` folder contains both unit tests as well as integrate tests and numerical experiments.


`archive_mlmc.sh` - for given directory compress or extract sampling subdirectory `output`
`common` - tests for the `src/common`
`mesh` - tests for mesh creation functions in `src/mesh`
`num_experiments` - numerical experiments with the whole system, input configurations and summary of the results, full results should be stored on metacentrum
`test-data` - various reference input data for the tests

`transport_vis.pvsm` - paraview pipeline for visualization of the transport model
`transport_vis.py` - same pipeline but in Python (could be used to modify view parametricaly)

All tests (with exception of GUI) should run in a Docker/Singularity image using possibly 
the [swrap](https://github.com/flow123d/swrap) tool.
