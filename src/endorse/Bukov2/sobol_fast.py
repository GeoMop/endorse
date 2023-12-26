import numpy as np

def separate_output_values(Y, D, N):
    #AB = np.zeros((N, D))
    #BA =  None
    #step =  D + 2
    YY = Y.reshape((*Y.shape[:-1], N, -1))
    A = YY[..., 0]      #Y[0 : Y.size : step]
    B = YY[..., D+1] #Y[(step - 1) : Y.size : step]
    AB = YY[..., 1:D+1]
    # for j in range(D):
    #     AB[:, j] = Y[(j + 1) : Y.size : step]
    #     if calc_second_order:
    #         BA[:, j] = Y[(j + 1 + D) : Y.size : step]
    return A, B, AB

def first_order(A, AB, B):
    """
    First order estimator following Saltelli et al. 2010 CPC, normalized by
    sample variance
    """
    y = np.r_[A, B]
    #if y.ptp() == 0:
    #    warn(CONST_RESULT_MSG)
    #    return np.array([0.0])

    return np.mean(B * (AB - A), axis=0) / np.var(y, axis=0)


def total_order(A, AB, B):
    """
    Total order estimator following Saltelli et al. 2010 CPC, normalized by
    sample variance
    """
    y = np.r_[A, B]
    return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(y, axis=0)


def _vec_sobol_total_only(Y, n_params, n_samples,
                         num_resamples=100, conf_level=0.95):
    """
    array shape (..., n_samples)
    :param array:
    :param problem:
    :return:
    """

    # determining if groups are defined and adjusting the number
    # of rows in the cross-sampled matrix accordingly
    D = n_params
    N = n_samples

    YY = Y.reshape((*Y.shape[:-1], N, -1))
    s1_samples = list(range(D+2))
    s1_samples[D+1] = 2*(D+1) - 1
    YY = YY[...,s1_samples]
    # Normalize the model output.
    # Estimates of the Sobol' indices can be biased for non-centered outputs
    # so we center here by normalizing with the standard deviation.
    # Other approaches opt to subtract the mean.
    mean_Y = YY.mean(axis=(-1, -2))
    std_Y = YY.std(axis=(-1, -2))
    YY = (YY - mean_Y[..., None, None]) / std_Y[..., None,None]
    A = YY[..., 0]
    B = YY[..., D+1]
    AB = YY[..., 1:D+1]

    y_con = np.concatenate((A, B), axis=-1)
    var_y = np.var(y_con, axis=-1)

    # Preliminary part of efficient S1 calculation
    #AB_diff = (AB[..., :, :] - A[..., :, None])
    #S_tot = B[..., None, :] @ AB_diff
    #S_tot = (S_tot.squeeze(axis=-2)) / n_samples / var_y

    # Total index
    S_tot = 0.5 * np.mean((AB[...,:,:] - A[...,:, None]) ** 2, axis=-2) / var_y[..., None]
    return S_tot[..., None]


def vec_sobol_total_only(array, problem):
    n_params = problem['num_vars']
    n_nested =  2 * (n_params + 1) if problem['second_order'] else n_params + 2

    sobol_fn = lambda x: _vec_sobol_total_only(x, n_samples=int(array.shape[-1] / n_nested), n_params=n_params)
    variant_samples = array.reshape((-1, array.shape[-1]))
    #variant_sobols = np.stack([sobol_fn(row) for row in variant_samples])
    variant_sobols = sobol_fn(variant_samples)

    return variant_sobols.reshape(*array.shape[:-1], *variant_sobols.shape[-2:])
