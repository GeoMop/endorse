import logging
from typing import *
import datetime
import joblib

import hashlib
from functools import wraps
import time
import os


"""
TODO: modify redis_simple_cache or our memoize decorator to hash also function code
 see https://stackoverflow.com/questions/18134087/how-do-i-check-if-a-python-function-changed-in-live-code
 that one should aslo hash called function .. the whole tree
 more over we should also hash over serialization of classes
"""

class EndorseCache:
    """
    Specific configuration of the joblib.Memory function call cache.
    - singleton
    - TODO: runt reduce_size in separate thread at the job start for
      at most 1min, in order to keep the cache functional.
    - TODO: log function calls: time, reuse

    In order to clear cached calls of a function use:
    @memoize
    def foo():
        pass

    # here clear all calls of foo
    foo.clear()
    """
    __instance__ = None
    @staticmethod
    def instance(*args, **kwargs):
        if EndorseCache.__instance__ is None:
            EndorseCache.__instance__ = EndorseCache(*args, **kwargs)
        return EndorseCache.__instance__

    def __init__(self, cachedir=None):
        if cachedir is None:
            # Get the home directory using os.environ and construct the path
            home_dir = os.environ['HOME']
            cachedir = os.path.join(home_dir, 'endorse_cache')
        self.memory = joblib.Memory(cachedir, verbose=0)

    def clear_all(self):
        """
        Clear the whole cache.
        """
        self.memory.clear()
        #self.cache.expire_all_in_set()


    def reduce_size(self,
                    bytes_limit="1000G",
                    items_limit="1000000",
                    age_limit=datetime.timedelta(days=365)):
        self.memory.reduce_size(bytes_limit, items_limit, age_limit)


# Workaround missing module in the function call key
# def memoize():
#     endorse_cache = EndorseCache.__instance__
#     def decorator(fn):
#         # redis-simple-cache does not include the function module into the key
#         # we poss in a functions with additional parameter
#         def key_fn(fn_id , *args, **kwargs):
#             return fn(*args, **kwargs)
#
#         modif_fn =  redis_cache.cache_it(limit=10000, expire=redis_cache.DEFAULT_EXPIRY, cache=endorse_cache.cache)(key_fn)
#
#         @wraps(fn)
#         def wrapper(*args, **kwargs):
#             return modif_fn((fn.__name__, fn.__module__), *args, **kwargs)
#         return wrapper
#     return decorator

def memoize(fn):
    """
    Decorator for memoizing endorse expensive functions.
    """
    endorse_cache = EndorseCache.instance()
    return endorse_cache.memory.cache(fn)




class File:
    """
    An object that should represent a file as a computation result.
    Contains the path and the file content hash.
    The system should also prevent modification of the files that are already created.
    To this end one has to use File.open instead of the standard open().
    Current usage:

    with File.open(path, "w") as f:
        f.write...

    return File.from_handle(f)  # check that handel was opened by File.open and is closed, performs hash.

    Ideally, the File class could operate as the file handle and context manager.
    However that means calling system open() and then modify its __exit__ method.
    However I was unable to do that. Seems like __exit__ is changed, but changed to the original one smowere latter as
    it is not called. Other possibility is to wrap standard file handle and use it like:

    @joblib.task
     def make_file(file_path, content):`
        with File.open(file_path, mode="w") as f: # calls self.handle = open(file_path, mode)
            f.handle.write(content)
        # called File.__exit__ which calls close(self.handle) and performs hashing.
        return f

    TODO: there is an (unsuccessful) effort to provide special handle for writting.
    TODO: Override deserialization in order to check that the file is unchanged.
          Seems that caching just returns the object without actuall checking.
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
    def __init__(self, path: str, files:List['File'] = None):  # , hash:Union[bytes, str]=None) #, postponed=False):
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
