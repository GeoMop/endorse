import numpy as np
import endorse.equivalent_field as ef


def test_from_voigt():
    eq_tn_fn = ef.eq_tensor(3, 'sym')
    # 3D
    voigt = [100, 10, 1, 11, 101, 110]
    result = eq_tn_fn.to_full_tn(voigt).reshape(3,3)
    assert np.allclose(result, [[100, 110, 101], [110, 10, 11], [101, 11, 1]])
    voigt = np.array(voigt)
    voigt_field = np.stack((voigt, 2* voigt))
    assert np.allclose(eq_tn_fn.to_full_tn(voigt_field), [[100, 110, 101, 110, 10, 11, 101, 11, 1], [200, 220, 202, 220, 20, 22, 202, 22, 2]])

    # 2D
    eq_tn_fn = ef.eq_tensor(2, 'sym')
    voigt = [10, 1, 11]
    result = eq_tn_fn.to_full_tn(voigt).reshape(2, 2)
    assert np.allclose(result, [[10, 11], [11, 1]])
    voigt = np.array(voigt)
    voigt_field = np.stack((voigt, 2* voigt))
    assert np.allclose(eq_tn_fn.to_full_tn(voigt_field), [[10, 11, 11, 1], [20, 22, 22, 2]])

def test_eq_tensor_3d():
    eq_tn_fn = ef.eq_tensor(3, 'sym')
    # without noise, eigen vectors parallel with axes
    loads = np.array([[1, 0, 0], [0, 1, 0], [0,0,1], [1, 1, 0]], dtype=float)
    K = np.array([[100, 0, 0], [0,10, 0], [0, 0, 1]], dtype=float)
    responses = loads @ K
    eq_tn_voigt = eq_tn_fn.flat(loads, responses)
    assert eq_tn_voigt.shape == (6,)
    eq_tn = eq_tn_fn.to_full_tn(eq_tn_voigt)
    #assert eq_tn_voigt.shape == (3, 3)
    assert np.allclose(eq_tn, K.flatten())

    # random loads
    np.random.seed(1)
    loads = np.random.random((3,3))
    responses = loads @ K
    eq_tn = eq_tn_fn.full(loads, responses)
    assert np.allclose(eq_tn, K.flatten())

    # with noise
    sigma = 0.01
    responses = loads @ K + np.random.normal(0.0, sigma, responses.shape)
    eq_tn = eq_tn_fn.full(loads, responses)
    assert np.allclose(eq_tn, K.flatten(), atol=0.1)

    # non-pos def
    # loads = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    # responses = np.array([[1e10, 0, 0], [0, -0.1, 0], [0, 0, 1]], dtype=float)
    # eq_tn_voigt = ef.eq_tensor_3d_sym(loads, responses)
    # assert np.allclose(eq_tn_voigt, [1e10,1-3, 1, 0, 0, 0])