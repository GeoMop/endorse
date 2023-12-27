import pickle

from endorse import common
import time
import h5py

def load_cfg(cfg_file):
    workdir = cfg_file.parent
    cfg = common.config.load_config(cfg_file)
    return workdir, cfg


class HDF5Files:
    def __init__(self, file_paths, mode):
        self.file_paths = file_paths
        self.mode = mode
        self.files = []

    def __enter__(self):
        for path in self.file_paths:
            self.files.append(h5py.File(path, self.mode))
        return self.files

    def __exit__(self, exc_type, exc_value, traceback):
        for file in self.files:
            file.close()




def pkl_write(workdir, data, name):
    with open(workdir / name, 'wb') as f:
        pickle.dump(data, f)


def pkl_read(workdir, name):
    try:
        with open(workdir / name, 'rb') as f:
            result = pickle.load(f)
    except Exception:
        result = None
    return result


# def memoize(func):
#     def wrapper(workdir, *args, **kwargs):
#         fname = f"{func.__name__}.pkl"
#         val = pkl_read(workdir, fname)
#         if val is None:
#             val = func(args, kwargs)
#             pkl_write(workdir, val, fname)
#         return val
#     return wrapper
#



def memoize(func):
    """
    Simple memoization function, no dependence on input, store into
    a file derived from the function name.
    :param func:
    :return:
    """
    def wrapper(workdir, *args, **kwargs):
        fname = f"{func.__name__}.pkl"
        val = pkl_read(workdir, fname)
        force = kwargs.pop('force', False)
        if force is True or val is None:
            print(f"Execute {func.__name__}  ...", end='')
            start = time.process_time_ns()
            val = func(workdir, *args, **kwargs)
            sec = (time.process_time_ns() - start) / 1e9
            print(f"[{sec}] s.")

            pkl_write(workdir, val, fname)
        else:
            print(f"Skip {func.__name__}.")
        return val
    return wrapper




def file_result(filename):
    """
    Reuse the file result.
    :param func:
    :return:
    """
    def decorator(func):
        def wrapper(workdir, *args, **kwargs):
            fname = workdir / filename
            if not fname.exists():
                print(f"Execute {filename} = {func.__name__}  ...", end='')
                start = time.process_time_ns()
                val = func(workdir, filename, *args, **kwargs)
                sec = (time.process_time_ns() - start) / 1e9
                print(f"[{sec}] s.")
            else:
                val = filename
                print(f"Skip {filename} = {func.__name__}.")
            return val

        return wrapper
    return decorator