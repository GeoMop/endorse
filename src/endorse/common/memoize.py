import logging
from typing import *
# import redis_cache
import pathlib
import joblib
import hashlib
from functools import wraps
import time
import os

# # 1) Grab joblib’s named logger and set its level
# joblib_logger = logging.getLogger("joblib")
# joblib_logger.setLevel(logging.INFO)
#
# # 2) Create a FileHandler that writes INFO+ only to joblib_cache.log
# joblib_handler = logging.FileHandler("joblib_cache.log", mode="a")
# joblib_handler.setLevel(logging.INFO)
#
# # 3) (Optional) If you want a custom format, attach a Formatter
# formatter = logging.Formatter(
#     "%(asctime)s %(name)s [%(levelname)s] %(message)s"
# )
# joblib_handler.setFormatter(formatter)
#
# # 4) Only push this handler onto the 'joblib' logger (not root!)
# joblib_logger.addHandler(joblib_handler)

"""
Caching of pure function calls currently based on the joblib.
- File wrapper class allows safe file results with appropriate hashes.

TODO: 
- support for other storage methods
- hashing of function implementation and subcalls 
  (working prototype in endorse-experiment, but does not generilize to more complex programs)
 see https://stackoverflow.com/questions/18134087/how-do-i-check-if-a-python-function-changed-in-live-code
 that one should aslo hash called function .. the whole tree
 more over we should also hash over serialization of classes

- program execution view in browser (? how related to Ray, Dask, ..)

"""


class CallCache:
    """
    Global singleton for the function call cache.
    Configuration is lazy: parameters passed to the instance method
    are stored and updated by subsequent instance calls, but the actual
    instance is created during first call of the memoized function.
    """
    __instance_args__ = {}
    __singleton_instance__ = None

    @staticmethod
    def __instance__():
        if CallCache.__singleton_instance__ is None:
            CallCache.__singleton_instance__ = CallCache(**CallCache.__instance_args__)
        return CallCache.__singleton_instance__

    @staticmethod
    def instance(**kwargs):
        """
        Parameters:
        workdir - str or Path where to place the cache
        expire_all - if True, delete whole cache

        parameters passed to joblib.Memory:
        verbose
        """
        CallCache.__instance_args__.update(kwargs)

    def __init__(self, workdir="", expire_all=False, **kwargs):
        # TODO: possibly start redis server
        self.workdir = pathlib.Path(workdir)

        self.mem_cache = joblib.Memory(
            location=self.workdir / "joblib_cache",
            **kwargs)

        if expire_all:
            self.mem_cache.clear()

    def expire_all(self):
        """
        Deprecated, call instance with 'expire_all=True' instead.
        """
        CallCache.instance(expire_all=True)


def memoize(fn):
    decorated_fn = None
    @wraps(fn)
    def wrapper(*args, **kwargs):
        nonlocal decorated_fn
        if decorated_fn is None:
            mem: joblib.Memory = CallCache.__instance__().mem_cache
            decorated_fn = mem.cache(fn)
        return decorated_fn(*args, **kwargs)
    return wrapper




class File:
    """
    An object that should represent a file as a computation result.
    Use cases:
    - pass File(file_path) to a memoize function that reads from a file
    - return File(file_path) from a memoize function that writes to a file

    Contains the path and the file content hash.
    The system should also prevent modification of the files that are already created.
    To this end one has to use File.open instead of the standard open().
    Current usage:

    with File.open(path, "w") as f:
        f.write...

    return File.from_handle(f)  # check that handel was opened by File.open and is closed, performs hash.

    Ideally, the File class could operate as the file handle and context manager.
    However that means calling system open() and then modify its __exit__ method.
    However I was unable to do that. Seems like __exit__ is changed, but changed to the original
    one smowere latter as
    it is not called. Other possibility is to wrap standard file handle and use it like:

    @joblib.task
     def make_file(file_path, content):`
        with File.open(file_path, mode="w") as f: # calls self.handle = open(file_path, mode)
            f.handle.write(content)
        # called File.__exit__ which calls close(self.handle) and performs hashing.
        return f
    """

    # @classmethod
    # def handle(cls, fhandle):
    #     return File(fhandle.name)

    # @classmethod
    # def output(cls, path):
    #     """
    #     Create File instance intended for write.
    #     The hash is computed after call close of the of open() handle.
    #     Path is checked to not exists yet.
    #     """
    #     return cls(path, postponed=True)
    _hash_fn = hashlib.md5

    def __init__(self, path: str, files: List['File'] = None):  # , hash:Union[bytes, str]=None) #, postponed=False):
        """
        For file 'path' create object containing both path and content hash.
        Optionaly the files referenced by the file 'path' could be passed by `files` argument
        in order to include their hashes.
        :param path: str
        :param files: List of referenced files.
        """
        self.path = os.path.abspath(path)
        if files is None:
            files = []
        self.referenced_files = files
        self._set_hash()

    def __getstate__(self):
        return (self.path, self.referenced_files)

    def __setstate__(self, args):
        self.path, self.referenced_files = args
        self._set_hash()

    def _set_hash(self):
        files = self.referenced_files
        md5 = self.hash_for_file(self.path)
        for f in files:
            md5.update(repr(f).encode())
        self.hash = md5.hexdigest()

    @staticmethod
    def open(path, mode="wt"):
        """
        Mode could only be 'wt' or 'wb', 'x' is added automaticaly.
        """
        exclusive_mode = {"w": "x", "wt": "xt", "wb": "xb"}[mode]
        # if os.path.isfile(path):
        #    raise ""
        fhandle = open(path, mode=exclusive_mode)  # always open for exclusive write
        return fhandle

    @classmethod
    def from_handle(cls, handle):
        assert handle.closed
        assert handle.mode.find("x") != -1
        return cls(handle.name)

    def __hash__(self):
        if self.hash is None:
            raise Exception("Missing hash of output file.")
        return hash(self.path, self.hash)

    def __str__(self):
        return f"File('{self.path}', hash={self.hash})"

    """
    Could be used from Python 3.11    
    @staticmethod
    def hash_for_file(path):
        with open(path, "rb") as f:
            return hashlib.file_digest(f, "md5")

        md5 = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(block_size), b''):
                md5.update(chunk)
        return md5.digest()
    """

    @staticmethod
    def hash_for_file(path):
        '''
        Block size directly depends on the block size of your filesystem
        to avoid performances issues
        Here I have blocks of 4096 octets (Default NTFS)
        '''
        block_size = 256 * 128
        md5 = File._hash_fn()
        try:
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(block_size), b''):
                    md5.update(chunk)
        except FileNotFoundError:
            raise FileNotFoundError(f"Missing cached file: {path}")
        return md5


"""


"""
