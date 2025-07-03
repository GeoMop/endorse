# Borehole Pressures Processing

Provides basic processing of the raw measurement data provided by SG-Geotechnika, all in 
the `input_data` dir. 

## Inputs

   - **`piezo*.xlsx`** continuous pressure data series in multi-packers, continuous measurement over WPTs and excavation.
   - **`wpt*.xlsx`** complementary WPT (water pressure tests) data with measured injected water mass
   - **`boreholes.yaml`** metadata about boreholes and *packer sections*
   - **`events.yaml`** eetadata about events affecting the measurement (todo: add wpt times)
The file `piezo_filtering.yaml` is meant to define different processing intervals,
not fixed yet.

## Sources overview
`decorators.py` - decorators for the processing functions (TODO: remove or move to endorse lib)
`piezo_canonic.py` - Processing of raw file into canonical flatten pandas table (denoised_df())

Running the module produces the overview plot (of both  original and denoised df)
and plots of individual BHs for the excavation period.

TODO:
1. add WPT events to events.yaml
2. generate Bayes report documents for individual sections, first page summary indicators:
   - best fit L1 norm + variance of the L1 norm
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

    
