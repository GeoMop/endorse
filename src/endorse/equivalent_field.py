"""
Functions for calculation of the equivalent scalar or tensor fields.
"""
from typing import *
from numpy import typing as npt
import numpy as np
import logging





EqPropertyFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
def eq_field(loads: npt.ArrayLike, responses: npt.ArrayLike, eq_fn: EqPropertyFn) -> np.ndarray:
    """
    loads - shape (N, K, D)
    responses - shape  (N, K, D)
    return: (N, DD)

    where:
    N - is number of field elements
    K - number of loads and reponses for a single result field element
    D - is dimension of loads and responses: 3/2 for equivalent tensors, missing for scalar homogenization
    DD - Voigt notation used for tensors;
         DD = 6 if D == 3
         DD = 3 if D == 2
         DD = None if D == 1 or missing
    """
    assert loads.shape == responses.shape and len(loads.shape) > 1
    loads, responses = np.atleast_3d(loads, responses)
    n_elements = loads.shape[0]
    dim = loads.shape[2]

    eq_field = np.empty((n_elements, voight_dim[dim]))
    for iel in range(n_elements):
        eq_field[iel, :] = eq_fn(loads[iel, :, :], responses[iel, :, :])
    return eq_field


class eq_tensor:
    @staticmethod
    def tn_homo_kernel(dim):
        homo_kernel = np.zeros((dim, dim, dim * dim))
        to_full_tn = np.empty((dim, dim), dtype=int)
        for i in range(dim):
            for j in range(dim):
                homo_kernel[i, j, dim * j + i] = 1
                to_full_tn[i, j] = dim * j + i
        return homo_kernel, to_full_tn.flatten()

    @staticmethod
    def tn_homo_kernel_sym(dim):
        ij_voigt = [
            [],
            [(0,0)],
            [(0, 0), (1, 1), (0, 1)],
            [(0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1)]
            ]
        voigt = ij_voigt[dim]
        homo_kernel_sym = np.zeros((dim, dim, len(voigt)))
        to_full_tn = np.empty((dim, dim), dtype=int)
        for k, (i, j) in enumerate(voigt):
            homo_kernel_sym[i, j, k] = 1.0
            homo_kernel_sym[j, i, k] = 1.0
            to_full_tn[i, j] = k
            to_full_tn[j, i] = k
        return homo_kernel_sym, to_full_tn.flatten()


    def __init__(self, dim, constrain=''):
        functions = {
            '': self.tn_homo_kernel,
            'sym': self.tn_homo_kernel_sym
        }
        self.dim = dim
        self.homo_kernel, self.to_full_indices = functions[constrain](dim)


    def flat(self, loads, responses):
        """
        loads - (K, 3)
        responses - (K, 3)
        returns - (6,) 3x3 symmetric pos-def tensor in Voigt notation

        """
        # form LS problem for 6 unknowns in Voigt notation: X, YY, ZZ, YZ, XZ, XY
        out_dim = self.homo_kernel.shape[2]  # dim of output tensor array
        ls_mat = (loads @ self.homo_kernel).reshape((-1, out_dim))
        rhs = responses.T.reshape((-1))

        result = np.linalg.lstsq(ls_mat, rhs, rcond=None)
        cond_tn, residuals, rank, singulars = result
        condition_number = singulars[0] / singulars[-1]
        if condition_number > 1e3:
            logging.warning(f"Badly conditioned inversion. Residual: {residuals}, max/min sing. : {condition_number}")
        return cond_tn

    def full(self, loads, responses):
        return self.to_full_tn(self.flat(loads, responses))


    def field_flat(self, loads, responses):
        return eq_field(loads, responses, self.flat)


    def field_full(self, loads, responses):
        return eq_field(loads, responses, self.full)

    def to_full_tn(self, eq_tn_flat: np.ndarray, axis:int = 1):
        """
        tn - an array with 3 or 6 elements along the `axis`
        Along given axis replace tensor in voigt notation by the flatten full DxD tensor.
        """
        tn_array = np.atleast_2d(eq_tn_flat)
        I = [slice(None)] * tn_array.ndim
        I[axis] = self.to_full_indices
        return tn_array[tuple(I)]





#def eq_tensor_3d_posdef(loads, responses):
#unit_loads = loads / np.linalg.norm(loads, axis=1)[:, None]
#load_components = np.sum(responses * unit_loads, axis=1)
# assert np.all(load_components > 0.0), f"{loads} : {responses}"
#responses_fixed = responses + (np.maximum(0, load_components) - load_components)[:, None] * unit_loads
#responses_fixed = responses
