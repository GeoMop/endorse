# "Chodby" project - experiment inversion

Application for processing pore pressure monitoring during and after excavation
and performing Bayes inversion to get hydraulic and mechanical rock properties.


## Install and environment
```
bash setup_env.sh
```

This will install the Python application `invapp` from sources in editable form (i.e. `pip -e <sources>`)
the application dependencies including the `endorse` generic code package are 
installed as well. 

## Structure

### [fitting](fitting/README.md)
Processing observed pressure data from multipacker.

### [input_data](input_data/README.md)

### hm_model 
Hydro-mechanical model of chamber pore pressures affected by the excavation.




## For developers

**Endorse** package should contain classes and functions with stable, documented API 
and relatively general usage.

**Steps**. Data processing steps could be developed in separate subdirectories as *callable* modules.
The structure is like:

```
def processing_fn(input_data):
    
    # some processing
    
    return output

def create_mock_data():
    return mock_data

if __name__ == "__main__":
    # individual step
    
    test_data = create_mock_data()
    result = processing_fn(test_data)
    
    # ... validate result
```

**Glueing** processing steps is possible through imports:
```
from invapp.mesh import processing_fn
```
**Functional Approach**
- Try to write **pure functions**, i.e. not modifying its inputs, 
  avoid methods modifying their objects.
- Use endorse.memoize (based on [joblib](https://github.com/joblib/joblib))
  instead of passing files. 
- Use small dataclasses, preferably using `attrs` library.
  If meangfull write methods that extract usefull data
- Prefer passing the class instance to creation of more complex structures
- After first working solution, go through and try to avoid code 
  duplicities, reuse the code.
- Write README.md in each directory. Summarize purpose of the code in that 
  dir in single sentence. Write single sentence to each file.
  If the file could not be described in singel sentence, 
  possibly good candidate to split it or do other code refactoring.
- Try to minimize dependencies. Put dataclasses into separate sources
  then plotting and meshing ...

**Documentation** 



# Tools for observation processing and Bayes inversion

## Common environment

```
bash setup_env
```

Installed packages given by `requirements.txt`.



**Fitting water pressure tests**
