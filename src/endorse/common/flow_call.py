from typing import *
import logging
import os
from . import dotdict, memoize, File, report, substitute_placeholders
import bp_simunek.common
import subprocess
import yaml
from pathlib import Path

def search_file(basename, extensions):
    """
    Return first found file or None.
    """
    if type(extensions) is str:
        extensions = (extensions,)
    for ext in extensions:
        if os.path.isfile(basename + ext):
            return File(basename + ext)
    return None

class EquationOutput:
    def __init__(self, eq_name, balance_name):
        self.eq_name: str = eq_name
        self.spatial_file: File = search_file(eq_name+"_fields", (".msh", ".pvd"))
        self.balance_file: File = search_file(balance_name+"_balance", (".yaml", ".txt"))
        self.observe_file: File = search_file(eq_name+"_observe", ".yaml")

    def _load_yaml_output(self, file, basename):
        if file is None:
            raise FileNotFoundError(f"Not found Flow123d output file: {self.eq_name}_{basename}.yaml.")
        with open(file.path, "r") as f:
            loaded_yaml = yaml.load(f, yaml.CSafeLoader)
        return dotdict.create(loaded_yaml)

    def observe_dict(self):
        return self._load_yaml_output(self.observe_file, 'observe')

    def balance_dict(self):
        return self._load_yaml_output(self.balance_file, 'balance')

    def balance_df(self):
        """
        create a dataframe for the Balance file
        rows for times, columns are tuple (region, value),
        values =[ flux,  flux_in,  flux_out,  mass,  source,  source_in,  source_out,  flux_increment,  source_increment,  flux_cumulative,  source_cumulative,  error ]
        :return:
        TODO: ...
        """
        dict = self.balance_dict()
        pass



class FlowOutput:

    def __init__(self, workdir: Path, process: subprocess.CompletedProcess):
        self.process = process
        output_dir = "output"
        with bp_simunek.common.workdir(str(workdir)):
            self.stdout = File("stdout")
            self.stderr = File("stderr")
        with bp_simunek.common.workdir(str(workdir/output_dir)):
            self.log = File("flow123.0.log")
            # TODO: flow ver 4.0 unify output file names
            self.hydro = EquationOutput("flow", "water")
            self.solute = EquationOutput("solute", "mass")
            self.mechanic = EquationOutput("mechanics", "mechanics")

    @property
    def success(self):
        return self.process.returncode == 0

    def check_conv_reasons(self):
        """
        Check correct convergence of the solver.
        Reports the divergence reason and returns false in case of divergence.
        """
        with open(self.log.path, "r") as f:
            for line in f:
                tokens = line.split(" ")
                try:
                    i = tokens.index('convergence')
                    if tokens[i + 1] == 'reason':
                        value = tokens[i + 2].rstrip(",")
                        conv_reason = int(value)
                        if conv_reason < 0:
                            print("Failed to converge: ", conv_reason)
                            return False
                except ValueError:
                    continue
        return True


def _prepare_inputs(workdir, file_in, params):
    in_dir, template = os.path.split(file_in.path)
    suffix = "_tmpl.yaml"
    assert template[-len(suffix):] == suffix
    filebase = template[:-len(suffix)]
    main_input = workdir/(filebase + ".yaml")
    main_input, used_params = substitute_placeholders(file_in.path, main_input, params)
    return main_input


def _flow_subprocess(arguments, main_input):
    arguments.append(main_input.path)
    logging.info("Flow123d running with: " + " ".join(arguments))

    stdout_path = "stdout"
    stderr_path = "stderr"
    with open(stdout_path, "w") as stdout:
        with open(stderr_path, "w") as stderr:
            completed_process = subprocess.run(arguments, stdout=stdout, stderr=stderr)
    return completed_process, File(stdout_path), File(stderr_path)


@report
def flow_call_with_check(workdir: Path, executable_list, input_template: File, params: Dict[str,str],
                         result_files:List[Path]=[], timeout=0):
    """
    Common call of `flow_call` and `flow_check`.
    """
    completed_process, stdout, stderr = flow_call(workdir, executable_list, input_template, params)
    res, fo = flow_check(workdir, completed_process, result_files, timeout)
    return res, fo


@report
def flow_call(workdir: Path, executable_list, input_template: File, params: Dict[str,str])\
        -> (subprocess.CompletedProcess, File, File):
    """
    Run Flow123d in actual work dir with main input given be given template and dictionary of parameters.

    1. change dir to workdir (resolve abs path)
    2. prepare the main input file from input_template file: suppose filename ends with + "_tmpl.yaml"
    3. run Flow123d

    Returns CompletedProcess (by subprocess), standard output, standard error output.

    TODO: pass only flow configuration
    """
    workdir_abs = workdir.absolute()
    orig_dir = os.getcwd()
    os.chdir(workdir)

    main_input = _prepare_inputs(workdir_abs, input_template, params)
    completed_process, stdout, stderr = _flow_subprocess(executable_list.copy(), main_input)
    logging.info(f"Flow123d exit status: {completed_process.returncode}")

    os.chdir(orig_dir)
    return completed_process, stdout, stderr


def flow_check(workdir: Path, completed_process, result_files:List[Path]=[], timeout=0) -> (bool, FlowOutput):
    """
    Check results of Flow123d, possibly output of `flow_call`.

    Create FlowOutput object.
    If any `result_files` are requested, check their existence.
        If they exist, then check Flow123d for convergence reason and return.
    Else check only the return code of subprocess(Flow123d) and return.

    TODO: wait for existence of output files in given timeout
    """
    res = False
    fo = FlowOutput(workdir, completed_process)

    # check the user requsted result files exist:
    if all([(workdir/f).exists() for f in result_files]):
        conv_check = fo.check_conv_reasons()
        logging.info(f"Flow123d convergence reason: {conv_check}")
        res = conv_check >= 0   # Flow123d algebraic solver converged
        return res, fo

    # if completed_process.returncode != 0:
    #     with open(fo.stderr.path, "r") as stderr:
    #         print(stderr.read())
    #     raise Exception("Flow123d ended with error")

    res = completed_process.returncode == 0
    return res, fo

# TODO:
# - call_flow variant with creating dir, copy,


