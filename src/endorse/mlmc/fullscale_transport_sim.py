import os
import copy
import logging
import shutil
import numpy as np
from typing import *
from endorse.fullscale_transport import transport_run, transport_2d, transport_result_format

import mlmc.random.correlated_field as cf
from typing import List
from mlmc.sim.simulation import Simulation
from mlmc.quantity.quantity_spec import QuantitySpec
from mlmc.level_simulation import LevelSimulation


class FullScaleTransportSim(Simulation):

    def __init__(self, cfg, mesh_steps):
        """
        :param config: Dict, simulation configuration
        """
        #super().__init__()
        self._config = cfg
        self._mesh_steps = mesh_steps

    def level_instance(self, fine_level_params: List[float], coarse_level_params: List[float]) -> LevelSimulation:
        """
        Called from mlmc.Sampler, it creates single instance of LevelSimulation (mlmc.level_simulation)
        :param fine_level_params: fine simulation step at particular level
        :param coarse_level_params: coarse simulation step at particular level
        :return: mlmc.LevelSimulation object
        """
        config = copy.deepcopy(self._config)
        # Set sample specific parameters
        # config["fine"] = {}
        # config["coarse"] = {}
        # config["fine"]["n_steps"] = fine_level_params[0]
        # config["coarse"]["n_steps"] = coarse_level_params[0]
        # config["res_format"] = self.result_format()

        return LevelSimulation(config_dict=config,
                               calculate=FullScaleTransportSim.calculate,
                               task_size=0.5,
                               need_sample_workspace=True)

    @staticmethod
    def calculate(config, seed):
        """
        Calculate fine and coarse sample and also extract their results
        :param config: general configuration
        :param seed: random number generator seed
        :return: np.ndarray, np.ndarray
        """

        #return np.zeros(285), np.zeros(285)
        from endorse import common
        from endorse.common import dotdict, memoize, File, call_flow, workdir, report
        from endorse.mesh_class import Mesh
          
        ###################
        ### fine sample ###
        ###################

        #conf_file = os.path.join(config["work_dir"], "test_data/config_homo_tsx.yaml")
        #cfg = common.load_config(conf_file)
        #cfg.flow_env["flow_executable"] = config["flow_executable"]
        #cfg["work_dir"] = config["work_dir"]
        model_fn = {2: transport_2d, 3:transport_run}
        val = model_fn[config._model_dim](config, seed)
        #q10 = list(val)
        #add_values = (10 - len(q10)) * [0.0]
        #q10.extend(add_values) #fixed_indicators[:len(ind_time_max)] = np.array(ind_time_max)
        logging.info(f"Sample result: {val}")
        res_fine = np.asarray(val)
        #fine_res = fo.hydro
        #res_fine = np.arange(10)
        #####################
        ### coarse sample ###
        #####################
        res_coarse = np.zeros_like(res_fine)

        return res_fine, res_coarse

    def result_format(self) -> List[QuantitySpec]:
        """
        Result format in order to cope with limitation of current format
        quantile: Array[n_quantiles]
        quantile_series: TimeSeries(times, Array[n_qunatiles])
         quntaile labels and exponents must be retrieved from Reult spec or indicator_set

        :return:
        TODO: modify MLMC to allow arbitatry Quantity type on the simulation output.
        Need to serialize the generic format spec.
        """
        result = transport_result_format(self._config)
        n_ind = len(result)
        times = result[0].times
        i_time = QuantitySpec(name="indicator_time", unit="y", shape=(n_ind, 1), times=[1], locations=['0'])
        indicator = QuantitySpec(name="indicator_conc", unit="g/m3", shape=(n_ind, 1), times=[1], locations=['0'])
        ind_series = QuantitySpec(name="indicator_series", unit="g/m3", shape=(n_ind, 1), times=times, locations=['0'])
        return [i_time, indicator, ind_series]
