# Borehole Pressures Processing

Provides basic processing of the raw measurement data provided by SG-Geotechnika, all in 
the `input_data` dir. Essential datasets are `piezo*.xlsx` continuous data series in multi-packers.
Complementary water pressure tests with measured injected water mass are covered by `wpt*.xlsx` files.
Metadata about boreholes and *packer sections* are in `boreholes.yaml`. 
Metadata about events affecting the measurement should be collected in `events.yaml`, 
blasts, injections, etc. 
The file `piezo_filtering.yaml` is meant to define different processing intervals,
not fixed yet.

## Sources overview
`decorators.py` - decorators for the processing functions (TODO: remove or move to endorse lib)
`piezo_canonic.py` - Processing of raw file into canonical flatten pandas table (denoised_df())

Running the module produces the overview plot (of both  original and denoised df)
and plots of individual BHs for the excavation period.

TODO:
Current filtering is better than nothing, but not great, if the local fitting appears viable, we should 
return to it and improve it.
2. better detection of essential jumps. Current approach compare jump with both sides, but 
   blast jumps are "constant" on the left and high variation on right. So usage of curvature
   (second derivative) seems to suit better. I.e. jump > threshold and left and right curvature
   is under other threshold. 
2. Good detection of the jumps allows then to mark noisy points and apply exponential filter
   (ewm) to the data or Gaussian kernel. 


## Jupyter stubs
Idea was to have self contained Jupyter conda environment for data processing. 
Not composible into larger project. Possible way is:
1. Processing functions in `measurement_processing.py`
2. Install whole app in jupyter conda env. -> Jupyter notebook presenting results of the processing.

`conda-requirements.yml` - configuration of conda environment for jupyter notebook
`create_env.sh` - creation of the conda environment
`endorse-fit.ipynb` - simple models and their fits, Jupyter notebook
`jupyter.sh` - starting jupyter lab with requested packages

    
