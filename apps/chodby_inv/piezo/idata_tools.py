import pickle
import os 

def save_idata_to_file(idata, filename):
    # if path doesn't exist, create it
    print(f"Saving idata {filename}...")

    if os.path.exists(filename):
        with open(filename, "wb") as file:
            pickle.dump(obj=idata, file=file)
    else:
        with open(filename, "ab") as file:
            pickle.dump(obj=idata, file=file)

def read_idata_from_file(filename):
    print(f"Reading idata from {filename}")
    try:
        with open(filename, "rb") as file:
            idata = pickle.load(file=file)
            return idata
    except:
        print("Error reading idata file")


def remove_burnin(idata, burnin=0):
    if burnin > 0:
        return idata.isel(draw=slice(burnin, None))
    
    if "burnin" in idata.attrs:
        return idata.isel(draw=slice(idata.attrs["burnin"], None))
    
    return idata