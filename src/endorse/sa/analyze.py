from SALib.analyze.fast import analyze as fast
from SALib.analyze.rbd_fast import analyze as rbd_fast
from SALib.analyze.morris import analyze as morris
from SALib.analyze.sobol import analyze as sobol
from SALib.analyze.delta import analyze as delta
from SALib.analyze.dgsm import analyze as dgsm
from SALib.analyze.ff import analyze as ff
from SALib.analyze.pawn import analyze as pawn
from SALib.analyze.hdmr import analyze as hdmr
#from SALib.analyze.rsa import analyze as rsa
#from SALib.analyze.discrepancy import analyze as discrepancy

import numpy as np

def sobol_indices_array_2(sobol_dict):
    indices_list = []
    indices_list.append(sobol_dict['ST'])
    indices_list.append(sobol_dict['ST_conf'])
    indices_list.append(sobol_dict['S1'])
    indices_list.append(sobol_dict['S1_conf'])
    sym_s2 = np.nan_to_num(sobol_dict['S2'])
    sym_s2 = sym_s2 + sym_s2.T
    sym_conf_s2 = np.nan_to_num(sobol_dict['S2_conf'])
    sym_conf_s2 = sym_conf_s2 + sym_conf_s2.T
    indices_list.extend([*sym_s2])
    indices_list.extend([*sym_conf_s2])
    return np.stack(indices_list, axis=1)



def sobol_indices_array_1(sobol_dict):
    indices_list = []
    indices_list.append(sobol_dict['ST'])
    indices_list.append(sobol_dict['ST_conf'])
    indices_list.append(sobol_dict['S1'])
    indices_list.append(sobol_dict['S1_conf'])
    return np.stack(indices_list, axis=1)

def sobol_vec(array, problem):
    """
    from (..., n_samples) -> (..., n_params, n_indices)
    :param array:
    :param problem:
    :param second_order:
    :return:
    """
    second_order = problem['second_order']
    #if second_order:
    #    sobol_vec = lambda sobol_dict : np.array(concatenate()
    if second_order:
        sobol_array_fn = sobol_indices_array_2
    else:
        sobol_array_fn = sobol_indices_array_1
    sobol_fn = lambda x : sobol_array_fn(sobol(problem, x, calc_second_order=second_order, print_to_console=False))
    variant_samples = array.reshape((-1, array.shape[-1]))
    variant_sobols = np.stack([sobol_fn(row) for row in variant_samples])
    return variant_sobols.reshape(*array.shape[:-1], *variant_sobols.shape[-2:])


def arr_to_dict(sobol_array):
    """
    Convert back from (n_params,n_incdices) to the Dict.
    :param sobol_array:
    :return:
    """
    n_params, n_indices = sobol_array.shape
    base = dict(
        ST=sobol_array[:, 0],
        ST_conf=sobol_array[:, 1],
        S1=sobol_array[:, 2],
        S1_conf=sobol_array[:, 3]
    )
    if n_indices > 4:
        base['S2'] = sobol_array[:,4:4+n_params]
        base['S2_conf'] = sobol_array[:, 4 + n_params:4 + 2 * n_params]
    return base

def sobol_max(array, axis=0, index = 0):
    #n_variants, n_param, n_indices = array.shape
    # return max over n_variants in given index
    # default is ST
    i_variants =  np.argmax(array[..., index], axis=axis, keepdims=True)
    i_variants = i_variants[..., None]
    amax_array = np.take_along_axis(array,i_variants, axis=axis).squeeze(axis=axis)
    return amax_array

