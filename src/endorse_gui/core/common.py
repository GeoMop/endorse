import logging
import os.path
from typing import *
import yaml
import subprocess
import shutil
from pathlib import Path
import numpy as np
from yamlinclude import YamlIncludeConstructor

from .memoize import File

class workdir:
    """
    Context manager for creation and usage of a workspace dir.

    name: the workspace directory
    inputs: list of files and directories to copy into the workspaceand
        TODO: fine a sort of robust ad portable reference
    clean: if true the workspace would be deleted at the end of the context manager.
    TODO: clean_before / clean_after
    TODO: File constructor taking current workdir environment, openning virtually copied files.
    portable reference and with lazy evaluation. Optional true copy possible.
    """
    CopyArgs = Union[str, Tuple[str, str]]
    def __init__(self, name:str="sandbox", inputs:List[CopyArgs] = None, clean=False):

        if inputs is None:
            inputs = []
        self._inputs = inputs
        self.work_dir = os.path.abspath(name)
        Path(self.work_dir).mkdir(parents=True, exist_ok=True)
        self._clean = clean
        self._orig_dir = os.getcwd()

    def copy(self, src, dest=None):
        """
        :param src: Realtive or absolute path.
        :param dest: Relative path with respect to work dir.
                    Default is the same as the relative source path,
                    for abs path it is the just the last name in the path.
        """
        if isinstance(src, File):
            src = src.path
        if isinstance(dest, File):
            dest = dest.path
        if dest is ".":
            if os.path.isabs(src):
                dest = os.path.basename(src)
            else:
                dest = src
        elif dest is None:
            dest = ""
        dest = os.path.join(self.work_dir, dest)
        dest_dir, _ = os.path.split(dest)
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        abs_src = os.path.abspath(src)
        if os.path.isdir(src):
            shutil.copytree(abs_src, dest, dirs_exist_ok=True)
        else:
            shutil.copy2(abs_src, dest)

    def __enter__(self):
        for item in self._inputs:
            if isinstance(item, Tuple):
                self.copy(*item)
            else:
                self.copy(item)
        os.chdir(self.work_dir)

        return self.work_dir

    def __exit__(self, type, value, traceback):
        os.chdir(self._orig_dir)
        if self._clean:
            shutil.rmtree(self.work_dir)


class dotdict(dict):
    """
    dot.notation access to dictionary attributes
    TODO: keep somehow reference to the original YAML in order to report better
    KeyError origin.
    """
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            return self.__getattribute__(item)

    @classmethod
    def create(cls, cfg : Any):
        """
        - recursively replace all dicts by the dotdict.
        """
        if isinstance(cfg, dict):
            items = ( (k, cls.create(v)) for k,v in cfg.items())
            return dotdict(items)
        elif isinstance(cfg, list):
            return [cls.create(i) for i in cfg]
        elif isinstance(cfg, tuple):
            return tuple([cls.create(i) for i in cfg])
        else:
            return cfg


def load_config(path):
    """
    Load configuration from given file replace, dictionaries by dotdict
    uses pyyaml-tags namely for:
    include tag:
        geometry: <% include(path="config_geometry.yaml")>
    """
    YamlIncludeConstructor.add_to_loader_class(loader_class=yaml.FullLoader, base_dir=os.path.dirname(path))
    with open(path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return dotdict.create(cfg)


def substitute_placeholders(file_in: str, file_out: str, params: Dict[str, Any]):
    """
    In the template `file_in` substitute the placeholders of format '<name>'
    according to the dict `params`. Write the result to `file_out`.
    """
    used_params = []
    with open(file_in, 'r') as src:
        text = src.read()
    for name, value in params.items():
        placeholder = '<%s>' % name
        n_repl = text.count(placeholder)
        if n_repl > 0:
            used_params.append(name)
            text = text.replace(placeholder, str(value))
    with open(file_out, 'w') as dst:
        dst.write(text)
    return File(file_out), used_params


# Directory for all flow123d main input templates.
# These are considered part of the software.
_script_dir = os.path.dirname(os.path.realpath(__file__))
flow123d_inputs_path = os.path.join(_script_dir, "../flow123d_inputs")

# TODO: running with stdout/ stderr capture, test for errors, log but only pass to the main in the case of
# true error


def sample_from_population(n_samples:int, frequency:Union[np.array, int]):
    if type(frequency) is int:
        frequency = np.full(len(frequency), 1, dtype=int)
    else:
        frequency = np.array(frequency, dtype=int)

    cumul_freq = np.cumsum(frequency)
    total_samples = np.sum(frequency)
    samples = np.random.randint(0, total_samples, size=n_samples + 1)
    samples[-1] = total_samples # stopper
    sample_seq = np.sort(samples)
    # put samples into bins given by cumul_freq
    bin_samples = np.empty_like(samples)
    i_sample = 0
    for ifreq, c_freq in enumerate(cumul_freq):

        while sample_seq[i_sample] < c_freq:
            bin_samples[i_sample] = ifreq
            i_sample += 1

    return bin_samples[:-1]