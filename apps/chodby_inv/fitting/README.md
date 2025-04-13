# Borehole Pressures Processing

Use `Chodby-inv` environment.

`measurement_processing.py` - Processing observed pressure data from multipacker.

1. convert to solumnd per chamber
2. introduce common simulation time
3. add chamber info
4. pressure jump detection
5. TODO: filtering
5. synchronise to blast timesdetect
6. 2D table, single measurement per row
7. plotting




## `measurement_data`
Folder with raw data for processing.

## `workdir`
Folder for preprocessed data.


## Jupyter stubs
Idea was to have selfcontained Jpyter conda environment for data processing. 
Not coposible into larger project. Possible way is:
1. Processing functions in `measurement_processing.py`
2. Install whole app in jupyter conda env. -> Jupyter notebook presenting results of the processing.

`conda-requirements.yml` - configuration of conda environment for jupyter notebook
`create_env.sh` - creation of the conda environment
`endorse-fit.ipynb` - simple models and their fits, Jupyter notebook
`jupyter.sh` - starting jupyter lab with requested packages

    
