# HM model

Hydro-mechanical models of excavation.

## 3D coupled model in Flow123d

- `boreholes.py`: Read and handle information about boreholes and measuring chambers (cells).
- `prepare_model_inputs.py`: Create necessary input files and data for the HM simulation.
- `run_model.py`: Main script for running either single Flow123d simulation or the initial pressure calibration.
- `visualize_fractures.py`: Functions for generating VTK files with fracture sets from geological mappings.