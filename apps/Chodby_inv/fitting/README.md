# Borehole Pressures Processing

## `measurement_data`
Folder with raw data for processing.

## `preprocessed`
Folder for preprocessed data.

`conda-requirements.yml` - configuration of conda environment for jupyter notebook

`create_env.sh` - creation of the conda environment

`endorse-fit.ipynb` - simple models and their fits, Jupyter notebook

`jupyter.sh` - starting jupyter lab with requested packages

`measurement_processing.py` - problem specific processing of raw measurement data. 
    - Raw data in plane 2D table
    - Merged with information about blasts and other operations.
    - Marked operation epochs (VTZ epochs, mining epoch)
    - Cleaned single sample drops, repaired data.
    - Common simulation time in days and thair fractions.
    
