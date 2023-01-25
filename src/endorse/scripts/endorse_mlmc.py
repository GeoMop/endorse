from typing import *
import os
import sys
import numpy as np
import attrs
import argparse
import fnmatch
import shutil
import time
import logging

logging.basicConfig(level=logging.INFO, filename='endorse_mlmc.log')
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

from concurrent.futures import ProcessPoolExecutor
import subprocess
import yaml
from glob import iglob

from endorse.mlmc.fullscale_transport_sim import FullScaleTransportSim

from mlmc.estimator import Estimate
from mlmc.sampler import Sampler
from mlmc.sampling_pool import OneProcessPool, ProcessPool
from mlmc.sampling_pool_pbs import SamplingPoolPBS
from mlmc.quantity.quantity import make_root_quantity
from mlmc.moments import Legendre
from mlmc.sample_storage_hdf import SampleStorageHDF
from mlmc.plot.plots import Distribution
from mlmc import estimator
from mlmc.quantity.quantity_estimate import estimate_mean, moments

from endorse import common
from endorse import plots

_script_dir = os.path.dirname(os.path.realpath(__file__))
_endorse_repository = os.path.abspath(os.path.join(_script_dir, '../../../'))

MAIN_CONFIG_FILE = 'config.yaml'

"""
tested parameters: run ../ --clean
"""

def create_sampling_pool(cfg_mlmc, work_dir, debug, max_n_proc=None):
        """
        Initialize sampling pool, object which
        :return: None
        """
        cfg_pbs = cfg_mlmc.get('pbs', None)
        if cfg_pbs is None:
            # separation of sampling is also good for GMSH which uses signals that can not be used within (non-main) threads
            return ProcessPool(max_n_proc, work_dir=work_dir, debug=debug)
            #return OneProcessPool(work_dir=work_dir, debug=debug)

        # Create PBS sampling pool
        sampling_pool = SamplingPoolPBS(work_dir=work_dir, debug=debug)
        sing_image = os.environ['SINGULARITY_CONTAINER']
        sing_bindings = os.environ['SINGULARITY_BIND']
        sing_venv = os.environ['SWRAP_SINGULARITY_VENV']
        #singularity_img = os.path.join(_endorse_repository, "tests/endorse_ci_7d9354.sif")
        pbs_config = dict(
            optional_pbs_requests=[],  # e.g. ['#PBS -m ae', ...]
            #home_dir="Why we need the home dir!! Should not be necessary.",
            #home_dir='/storage/liberec3-tul/home/martin_spetlik/',
            python=f'singularity exec  {sing_image} {sing_venv}/bin/python',
            #python='singularity exec {} /usr/bin/python3'.format(self.singularity_path),
            env_setting=[#'cd $MLMC_WORKDIR',
                         "export SINGULARITY_TMPDIR=$SCRATCHDIR",
                         "export PIP_IGNORE_INSTALLED=0",
                         'cd {}'.format(_endorse_repository),
                         #'singularity exec {} ./setup.sh'.format(singularity_img),
                         #'singularity exec {} venv/bin/python3 -m pip install scikit-learn'.format(singularity_img)
                         #'module load python/3.8.0-gcc',
                         #'source env/bin/activate',
                         #'module use /storage/praha1/home/jan-hybs/modules',
                         #'module load flow123d',
                         #'module unload python-3.6.2-gcc',
                         #'module unload python36-modules-gcc'
                         ],
            scratch_dir=None
        )
        pbs_config.update(cfg_pbs)
        sampling_pool.pbs_common_setting(flow_3=True, **pbs_config)

        return sampling_pool

@attrs.define
class SamplingParams:
    sample_sleep: float = 30,
    # self.init_sample_timeout = 600
    sample_timeout: float = 60,
    adding_samples_coef: float = 0.1

def create_sampling_params(workdir):
    """
    Set pbs config, flow123d, gmsh
    :return: None
    """
    root_dir = os.path.abspath(workdir)
    while root_dir != '/':
        root_dir, tail = os.path.split(root_dir)

    if tail == 'storage' or tail == 'auto':
        # Metacentrum
        return SamplingParams(
            sample_sleep = 10,
            #self.init_sample_timeout = 600
            sample_timeout = 5, # force loop only in all_collect with proper logging
            adding_samples_coef = 0.1
        )
    else:
        return SamplingParams(
            sample_sleep = 1,
            #self.init_sample_timeout = 600
            sample_timeout = 0.5,
            adding_samples_coef = 0.1
        )

def create_level_params(cfg_mlmc):
    step_range = [25, 5]  # @TODO: set meaningful values or specify level parameters in a different way.
    return estimator.determine_level_parameters(cfg_mlmc.n_levels, step_range)


def create_sampler(cfg, work_dir, debug, n_proc):
    """
    Simulation dependent configuration
    :return: mlmc.sampler instance
    """
    sampling_pool = create_sampling_pool(cfg.machine_config, work_dir, debug, max_n_proc=n_proc)
    level_parameters = create_level_params(cfg.mlmc)
    # General simulation config
    # conf_file = os.path.join(self.work_dir, "test_data/config_homogenisation.yaml")
    # cfg = self.load_config(conf_file)
    mesh_steps={level_parameters[0][0]: 50} # @TODO: check values
    #config['work_dir'] = self.work_dir
    #config["flow_executable"] = ["flow123d"]
    #config['source_params'] = dict(position=10, length=6)

    # Create simulation factory, instance of class that inherits from mlmc.sim.simulation
    simulation_factory = FullScaleTransportSim(cfg, mesh_steps)

    # Create HDF sample storage
    logging.info(f"[{work_dir}] Creating HDF storage: {work_dir}/mlmc_1.hdf5")
    sample_storage = SampleStorageHDF(file_path=os.path.join(work_dir, f"mlmc_{cfg.mlmc.n_levels}.hdf5"))

    # Create sampler, it manages sample scheduling and so on
    logging.info(f"[{work_dir}] Creating sampler ...")    
    sampler = Sampler(
        sample_storage=sample_storage,
        sampling_pool=sampling_pool,
        sim_factory=simulation_factory,
        level_parameters=level_parameters)
    logging.info(f"[{work_dir}] sampler done.")    

    return sampler

def all_collect(sampling_params, sampler, work_dir):
    """
    Collect samples
    :param sampler: mlmc.Sampler object
    :return: None
    """
    running = 1
    while running > 0:
        running = sampler.ask_sampling_pool_for_samples(
            sleep=sampling_params.sample_sleep,
            timeout=sampling_params.sample_timeout)
        logging.info(f"[{work_dir}] N running: {running}")


def run_fixed(cfg, n_samples, debug, n_proc):
    """
    Run MLMC.
    Fixed number of samples.
    :return: None
    """
    
    work_dir = os.path.abspath(".")
    sampling_params = create_sampling_params(work_dir)
    sampler = create_sampler(cfg, work_dir, debug, n_proc)
    
    running = sampler.ask_sampling_pool_for_samples(
        sleep=sampling_params.sample_sleep,
        timeout=sampling_params.sample_timeout)    
    logging.info(f"[{work_dir}] init N running: {running}")
    
    sampler.set_initial_n_samples(n_samples)
    sampler.schedule_samples()
    #sampler.ask_sampling_pool_for_samples(sleep=self.sample_sleep, timeout=self.sample_timeout)
    all_collect(sampling_params, sampler, work_dir)
    time.sleep(30) # workaround for the uncompleted HDF5 output



# class FullScaleTransport:
#
#     def __init__(self, cfg, debug):
#
#         self.work_dir = os.path.abspath(".")
#         #cfg = common.load_config(os.path.join(main_cfg_file))
#         self.cfg = cfg
#
#         #cfg.flow_env.mlmc.singularity
#         #self.singularity_image = os.path.abspath(cfg.mlmc.singularity_image)
#         #self.endorse_repository = os.path.abspath(args.endorse_dir)
#         # Add samples to existing ones
#         # Remove HDF5 file, start from scratch
#         self.debug = debug
#         # 'Debug' mode is on - keep sample directories
#         self.n_levels = cfg.mlmc.n_levels
#         self.n_moments = 9
#         self._quantile = 1e-3
#         self.sampling_params = create_sampling_params(self.work_dir)
#
#
#
#
#     def recollect(self):
#         self.run(recollect=True)
#
#     def renew(self):
#         sampler = create_sampler(self.cfg, self.work_dir, self.debug)
#         sampling_params = create_sampling_params(self.work_dir)
#         sampler.ask_sampling_pool_for_samples()
#         sampler.renew_failed_samples()
#         sampler.ask_sampling_pool_for_samples(sleep=sampling_params.sample_sleep, timeout=sampling_params.sample_timeout)
#         self.all_collect(sampler)  # Check if all samples are finished
#
#     def generate_jobs(self, sampler, target_var=None):
#         """
#         Generate level samples
#         :param n_samples: None or list, number of samples for each level
#         :param renew: rerun failed samples with same random seed (= same sample id)
#         :return: None
#         """
#         sampler.set_initial_n_samples()
#
#         if target_var is None:
#             return
#
#         root_quantity = make_root_quantity(storage=sampler.sample_storage,
#                                            q_specs=sampler.sample_storage.load_result_format())
#
#         moments_fn = self.set_moments(root_quantity, sampler.sample_storage, n_moments=self.n_moments)
#         estimate_obj = estimator.Estimate(root_quantity, sample_storage=sampler.sample_storage,
#                                           moments_fn=moments_fn)
#
#         # New estimation according to already finished samples
#         variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
#         n_estimated = estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
#                                                                        n_levels=sampler.n_levels)
#
#         # Loop until number of estimated samples is greater than the number of scheduled samples
#         while not sampler.process_adding_samples(n_estimated, self.sample_sleep, self.adding_samples_coef,
#                                                  timeout=self.sample_timeout):
#             # New estimation according to already finished samples
#             variances, n_ops = estimate_obj.estimate_diff_vars_regression(sampler._n_scheduled_samples)
#             n_estimated = estimator.estimate_n_samples_for_target_variance(target_var, variances, n_ops,
#                                                                                    n_levels=sampler.n_levels)
#
#     def set_moments(self, quantity, sample_storage, n_moments=5):
#         true_domain = estimator.Estimate.estimate_domain(quantity, sample_storage, quantile=0.01)
#         return Legendre(n_moments, true_domain)
#
#
#     # def process(self):
#     #     sample_storage = SampleStorageHDF(file_path=os.path.join(self.work_dir, "mlmc_{}.hdf5".format(self.n_levels)))
#     #     sample_storage.chunk_size = 1024
#     #     result_format = sample_storage.load_result_format()
#     #     root_quantity = make_root_quantity(sample_storage, result_format)
#     #
#     #     conductivity = root_quantity['indicator_conc']
#     #     time = conductivity[1]  # times: [1]
#     #     location = time['0']  # locations: ['0']
#     #     values = location[0, 0]  # result shape: (10, 1)
#     #
#     #     # Create estimator for quantities
#     #     x_estimator = self.create_estimator(values, sample_storage)
#     #
#     #     root_quantity_estimated_domain = Estimate.estimate_domain(root_quantity, sample_storage, self._quantile)
#     #     root_quantity_moments_fn = Legendre(self.n_moments, root_quantity_estimated_domain)
#     #
#     #     # There is another possible approach to calculating all moments at once and then choose quantity
#     #     moments_quantity = moments(root_quantity, moments_fn=root_quantity_moments_fn, mom_at_bottom=True)
#     #     moments_mean = estimate_mean(moments_quantity)
#     #     conductivity = root_quantity['conductivity']
#     #     time = conductivity[1]  # times: [1]
#     #     location = time['0']  # locations: ['0']
#     #     value_moments = location[0, 0]  # result shape: (1, 1)
#     #
#     #     # true_domain = [-10, 10]  # keep all values on the original domain
#     #     # central_moments_fn = Monomial(n_moments, true_domain, ref_domain=true_domain, mean=moments_mean())
#     #     # central_moments_quantity = moments(root_quantity, moments_fn=central_moments_fn, mom_at_bottom=True)
#     #     # central_moments_mean = estimate_mean(central_moments_quantity)
#     #     # print("central moments mean ", central_moments_mean())
#     #
#     #     FullScaleTransport._approx_distribution(x_estimator, self.n_levels, tol=1e-8)
#
#     def create_estimator(self, quantity, sample_storage):
#         estimated_domain = Estimate.estimate_domain(quantity, sample_storage, quantile=self._quantile)
#         moments_fn = Legendre(self.n_moments, estimated_domain)
#         # Create estimator for your quantity
#         return Estimate(quantity=quantity, sample_storage=sample_storage,
#                                               moments_fn=moments_fn)
#
#     @staticmethod
#     def _approx_distribution(estimator, n_levels, tol=1.95):
#         """
#         Probability density function approximation
#         :param estimator: mlmc.estimator.Estimate instance, it contains quantity for which the density is approximated
#         :param n_levels: int, number of MLMC levels
#         :param tol: Tolerance of the fitting problem, with account for variances in moments.
#         :return: None
#         """
#         distr_obj, result, _, _ = estimator.construct_density(tol=tol)
#         distr_plot = Distribution(title="distributions", error_plot=None)
#         distr_plot.add_distribution(distr_obj)
#
#         if n_levels == 1:
#             samples = estimator.get_level_samples(level_id=0)[..., 0]
#             distr_plot.add_raw_samples(np.squeeze(samples))
#         distr_plot.show(None)
#         distr_plot.reset()
#
#     @staticmethod
#     def determine_level_parameters(n_levels, step_range):
#         """
#         Determine level parameters,
#         In this case, a step of fine simulation at each level
#         :param n_levels: number of MLMC levels
#         :param step_range: simulation step range
#         :return: list of lists
#         """
#         assert step_range[0] > step_range[1]
#         level_parameters = []
#         for i_level in range(n_levels):
#             if n_levels == 1:
#                 level_param = 1
#             else:
#                 level_param = i_level / (n_levels - 1)
#             level_parameters.append([step_range[0] ** (1 - level_param) * step_range[1] ** level_param])
#         return level_parameters

CaseName = NewType('CaseName', str)
CasePatch = common.config.VariantPatch

@attrs.define
class SourceDensity:
    """
    For odd length N the density is (1/N, ..., 1/N),  N items
    for even length N the density is (1/2N, 1/N, ..., 1/N, 1/2N) N + 1 items
    """
    # first container indexed from 0
    center: int
    # number of containers in the source
    length: int

    def plot_label(self):
        #return f"pos: {self.center}, len: {self.length}"
        return f"{self.center}"

    def fs_name_items(self):
        c_str =  f"{self.center:03d}"
        if self.length > 1:
            return [c_str, str(self.source.length)]
        else:
            return [c_str]

    #@staticmethod
    #def from_center(center: int, length: int):
    #    return SourceDensity(center - , length)

@attrs.define
class SimCase:
    cfg: common.dotdict = attrs.field(repr=False)
    case_name: str
    case_patch: CasePatch
    source: SourceDensity
    _storage: Any = None

    @property
    def directory(self):
        items = [self.case_name, *self.source.fs_name_items()]
        return "-".join(items)

    @property
    def hdf5_path(self):
        abs_hdf = os.path.abspath(os.path.join(self.directory, "mlmc_1.hdf5"))
        print(abs_hdf)
        return abs_hdf

    @property
    def storage(self):
        if self._storage is None:
            self._storage = SampleStorageHDF(file_path=self.hdf5_path)
            self._storage.chunk_size = 1024

        return self._storage


    def root_quantity(self):
        logging.info(f"Getting  values from: {self.directory}")
        result_format = self.storage.load_result_format()
        logging.info(f"Result format: {result_format}")
        return make_root_quantity(self.storage, result_format)

    def log_inditator_quantity(self, i_quantile=1):
        root_quantity = self.root_quantity()
        ind_conc = root_quantity['indicator_conc']
        time = ind_conc[1]  # times: [1]
        location = time['0']  # locations: ['0']
        values = location[i_quantile, 0]  # selected quantile
        values = np.log10(values)
        # assert np.shape(values) == (1, 1)
        return values

    def time_quantity(self, i_quantile=1):
        root_quantity = self.root_quantity()
        ind_conc = root_quantity['indicator_time']
        time = ind_conc[1]  # times: [1]
        location = time['0']  # locations: ['0']
        values = location[i_quantile, 0]  # selected quantile
        # assert np.shape(values) == (1, 1)
        return values


    def mean_std_log(self):
        root_quantity = self.root_quantity()
        i_quantile = 1
        ind_conc = root_quantity['indicator_conc']
        time = ind_conc[1]  # times: [1]
        location = time['0']  # locations: ['0']
        values = location[i_quantile, 0]  # selected quantile
        values = np.log(values)
        assert values.shape[0] == 1
        values = values[0]
        #values = values.select(values < 1e-1)
        samples = self._get_samples(values, self.storage)[0, 0]


        q_mean = estimate_mean(values)
        val_squares = estimate_mean(np.power(values - q_mean.mean, 2))
        std = np.sqrt(val_squares.mean)

        return q_mean.mean[0], std[0], samples

    def _get_samples(self, quantity):
        #n_moments = 2
        #estimated_domain = Estimate.estimate_domain(quantity, sample_storage, quantile=0.001)
        #moments_fn = Legendre(n_moments, estimated_domain)
        #estimator = Estimate(quantity=quantity, sample_storage=sample_storage, moments_fn=moments_fn)
        estimator = Estimate(quantity=quantity, sample_storage=self.storage)
        samples = estimator.get_level_samples(level_id=0, n_samples=100)
        return samples[0,:, 0] # not clear why it has still shape (1, N, 1)

    def log_indicator_mc_samples(self, i_quantile=1):
        conc_quantity = self.log_inditator_quantity(i_quantile)
        time_quantity = self.time_quantity(i_quantile)

        return (self.case_name, self.source.plot_label(), self._get_samples(time_quantity), self._get_samples(conc_quantity))

    def clean(self, all=False):
        try:
            if all:
                shutil.rmtree(self.directory, ignore_errors=False, onerror=None)
            else:
                os.remove(os.path.join(self.directory, f"mlmc_{self.cfg.mlmc.n_levels}.hdf5"))
        except FileNotFoundError:
            pass


def comma_list(arg_list:str):
    return arg_list.split(',')

@attrs.define
class SimCases:
    """
    Organize set of stochastic simulation variants.
    That consists of cartesian product of variants (cases) of forward simulation parameters
    and distribution of contamination source (sources), currently just as discrete containers.

    Set of all named configuration cases is defined by 'cases.yaml' (file must be referenced in main 'config.yaml') that
    contains dictionary {<case_name> : <dict of config changes>}.

    Source distribution should be considered as prescribed set of concentration sources, i.s. concentration source density C(x)
    is sum of densities for individual containers. The diffusion through the bentonite would be described by a time changing
    and random parameter however the distribution is independent on the container position.
    This way the process is linear with respect to the concentration density and better describes differences between
    compared configurations.

    To this end we consider source given by its position and span (how many source containers we consider) the concentration is normalized
    so that total initial concentration is 1. Time decrease is part of configuration (see cases).
    <i> = [i:i+1:1]
    <i>-<n> = [i-n/2 : i+n/2+(1) :1]
    """

    cfg: common.dotdict
    cfg_cases : Dict[CaseName, CasePatch]
    source_densities: List[SourceDensity]

    @staticmethod
    def def_args(parser):
        help = """
               Space separated names of cases. Subset of cases defined as keys of `cases.yaml`."
               """
        parser.add_argument("cases", type=comma_list, help=help)
        help = '''
        Defines basis functions of the linear space of source densities.Space separated index ranges. E.g. "1:10:2 2 6:8"
        '''
        parser.add_argument("sources", type=comma_list, help=help)

    @classmethod
    def initialize(cls, args):
        """
        Initialize from cfg._cases file and arguments.
        :param args:
        :return:
        """
        cfg = common.load_config(MAIN_CONFIG_FILE)
        cases = cls.active_cases(args.cases, cfg.cases_file)
        sources = cls.source_basis(args.sources, cfg.geometry.containers.n_containers)
        return cls(cfg, cases, sources)

    @staticmethod
    def active_cases(arg_cases, cases_file):
        cfg = common.load_config(cases_file)
        if not isinstance(cfg, dict):
            cfg = {}
        all_cases = list(cfg.keys())
        selected_cases = []

        for case_pattern in arg_cases:
            selected_cases.extend(fnmatch.filter(all_cases, case_pattern))
        return {k:cfg[k] for k in selected_cases}

    @staticmethod
    def source_basis(arg_sources, n_containers):
        sources = []
        all_containers = list(range(n_containers))
        for slice_token in arg_sources:
            slice_items = [int(x.strip()) if x.strip() else None for x in slice_token.split(':')]
            if len(slice_items) == 1:
                # single number interpreted as single index contrary to the standard slice
                source_slice = slice(slice_items[0], slice_items[0] + 1, 1)
            else:
                source_slice = slice(*slice_items)
            subset = all_containers[source_slice]
            if not subset:
                continue
            step = subset[1] - subset[0] if len(subset) > 1 else 1
            for i in subset:
                sources.append(SourceDensity(i, step))
        return sources

    def iterate(self):
        for case_key, case_patch in self.cfg_cases.items():
            for source in self.source_densities:
                yield SimCase(self.cfg, case_key, case_patch, source)

    def mc_plots(self, label):
        data = [case.log_indicator_mc_samples(i_quantile=1) for case in self.iterate()]
        #print(data)
        plots.plot_mc_cases(data, 'log10 conc ' + r'$[g/m^3]$', label)
        #plots.indicator_timefunc(data,

    def mlmc_plots(self):
        data = [(case.case_name, case.source.plot_label(), *case.mean_std_log()) for case in self.iterate()]
        #print(data)
        plots.plot_log_errorbar_groups(data, 'conc ' + r'$[g/m^3]$')
        #plots.indicator_timefunc(data,

class CleanCmd:
    @staticmethod
    def def_args(parser):
        help="Remove work directories of given cases."
        parser.add_argument('--all', action='store_true', help=help)
        SimCases.def_args(parser)

    # Remove HFD5 file
    def execute(self, args):
        cases = SimCases.initialize(args)
        for case in cases.iterate():
            case.clean(args.all)


class PackCmd:
    @staticmethod
    def def_args(parser):
        help = \
        """
        Pack sampling results important for further processing.
        Created tar archive can be moved to other system and preserves 
        correct directory structure after extraction.
        Contains: config files, hdf5 files, failed and successfull samples without *.vtu, *.msh* files. 
        """
        parser.add_argument('--all', action='store_true', help="Include *.vtu and *.msh* files.")
        SimCases.def_args(parser)

    # Remove HFD5 file
    def execute(self, args):
        cwd = os.path.basename(os.getcwd())
        logging.info(f"Pack in {os.getcwd()}")
        cases = SimCases.initialize(args)
        cases_dirs = list({f"{cwd}/{case.directory}" for case in cases.iterate()})
            
        # add jobs/collected samples
        exclude = ['*/large_model_local.msh2', '*/output/jobs/*']
        if not args.all:
            exclude.extend(['*.vtu', '*.msh*'])
        main_wd_files=[f"{cwd}/{f}" for p in ["*.yaml", "*.pdf"] for f in iglob(p)]   
        exclude_args = [f"--exclude={p}" for p in exclude]    
        tar_command = ['tar', '-C', '..', '-cvzf', f"../{cwd}.tar.gz", *exclude_args, *main_wd_files, *cases_dirs]
        subprocess.run(tar_command)
        


@attrs.define
class RunCmd:
    @staticmethod
    def def_args(parser):
        parser.add_argument("-nt", "--n_thread", default=6, type=int,
                        help="Number of sampling threads, sampling cases in parallel.")
        parser.add_argument("-np", "--n_proc", default=2, type=int,
                        help="Number of processes per thread.")
        parser.add_argument("-d", "--debug", default=False, action='store_true',
                        help="Keep sample directories")
        parser.add_argument("-c", "--clean", default=False, action='store_true',
                        help="Remove previous HDF5 storage.")
        parser.add_argument("--dim", default=3, choices=[2,3], type=int,
                        help="Model dimension (2,3) for testing purpose.")
        SimCases.def_args(parser)

    def execute(self, args):
        #if args.clean:
        #    common.EndorseCache.instance().expire_all()
        logging.info(f"Main CWD: {os.getcwd()}")
        base_dir = os.getcwd()
        futures = []
        cases = SimCases.initialize(args)
        with ProcessPoolExecutor(max_workers = args.n_thread) as pool:
            for case in cases.iterate():
                if args.clean:
                    case.clean()
                f = pool.submit(self.run_case, base_dir, case, args.dim, args.n_proc, args.debug)
                futures.append((f, case))
        for f, case in futures:
            print(case, "result: ", f.result())

    @staticmethod
    def run_case(base_dir:str, case : SimCase, model_dim, np, debug):
        #print("running case:", case)
        logging.info(f"[{case.directory}] Creating thread..")
        #logging.info(f"{os.environ}")
        hostname = os.environ.get('ENDORSE_HOSTNAME', None)
        cfg = common.load_config(MAIN_CONFIG_FILE, collect_files=True, hostname=hostname)
        cfg_var = common.config.apply_variant(cfg, case.case_patch)
        cfg_var.transport_fullscale.source_params.source_ipos = case.source.center
        inputs = cfg._file_refs
        os.chdir(base_dir)
        logging.info(f"[{case.directory}] CWD: {os.getcwd()}")
        with common.workdir(case.directory, inputs=inputs):
            time.sleep(30)
            logging.info(f"[{case.directory}] CWD in the case dir: {os.getcwd()}")
            common.dump_config(cfg_var)
            #cfg_file = "cfg_variant.yaml"
            #with open(cfg_file, "w") as f:
            #        yaml.dump(common.dotdict.serialize(cfg_var), f)
            n_samples = cfg.mlmc.n_samples
            cfg_var._model_dim = model_dim
            run_fixed(cfg_var, n_samples, debug, n_proc=np)

class MC_CasesPlot:

    @staticmethod
    def def_args(parser):
        SimCases.def_args(parser)

    def execute(self, args):
        cases = SimCases.initialize(args)
        path, basename = os.path.split(os.getcwd())
        cases.mc_plots(basename)

@attrs.define
class PlotCmd:
    @staticmethod
    def def_args(parser):
        add_subparsers(parser, 'plot', 'plot_class', [MC_CasesPlot])

    def execute(self, args):
        plot_instance = args.plot_class()
        plot_instance.execute(args)



def add_subparsers(parser:argparse.ArgumentParser, suffix, fn_arg:str, classes: List[Any]):
    subparsers = parser.add_subparsers()
    for cls in classes:
        cmd = cls.__name__.lower().rstrip(suffix)
        subparser = subparsers.add_parser(cmd)
        cls.def_args(subparser)
        subparser.set_defaults(**{fn_arg: cls})

def get_arguments(arguments):
    """
    Getting arguments from console
    :param arguments: list of arguments
    :return: namespace
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workdir',
                        default=os.getcwd(),
                        type=str, help='Main directory of the whole project. Default is current directory.')
    add_subparsers(parser, 'cmd', 'cmd_class', [CleanCmd, RunCmd, PlotCmd, PackCmd])
    args = parser.parse_args(arguments)
    return args

def main():
    args = get_arguments(sys.argv[1:])
    with common.workdir(args.workdir):
        time.sleep(30)
        command_instance = args.cmd_class()
        command_instance.execute(args)

if __name__ == "__main__":
    main()
    

