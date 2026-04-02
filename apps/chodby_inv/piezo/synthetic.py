import numpy as np
import arviz as az
import xarray as xr
from ray import logger
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from . import save_idata_to_file

#az.rcParams["plot.max_subplots"] = 100

import tinyDA as tda
from chodby_inv.piezo.plots import plot_likelihood, plot_trace_modified, save_plots_pdf_pages

class Model:
    """
    Base class for synthetic models.
    """
    def evaluate(self, x: np.ndarray) -> np.array:
        """
        Evaluate the model at a given point x.
        
        :param x: Input array representing the point at which to evaluate the function.
        :type x: np.ndarray
        :return: The value of the function at the given point x.
        :rtype: float
        """
        raise NotImplementedError("Subclasses should implement this method.")

class AckleyModel(Model):
    """
    Ackley function model for synthetic testing.
    """
    def __init__(
            self,
            a=20,
            b=0.2,
            c=2*np.pi,
            d=10
        ):
        """
        Initialize the Ackley function model.
        
        :param a: Meta parameter a.
        :param b: Meta parameter b.
        :param c: Meta parameter c.
        """
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __str__(self):
        print_a = "{:.2f}".format(self.a)
        print_b = "{:.2f}".format(self.b)
        print_c = "{:.2f}".format(self.c)
        return f"AckleyModel(a={print_a}, b={print_b}, c={print_c})"

    def evaluate(self, x: np.ndarray) -> np.array:
        """
        Evaluate the Ackley function at a given point x.
        
        :param x: Input array representing the point at which to evaluate the function.
        :type x: np.ndarray
        :return: The value of the Ackley function at the given point x.
        :rtype: float
        """
        d = len(x)
        assert d == self.d, "Input dimension must match the model dimension."
        
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return np.array([term1 + term2 + self.a + np.exp(1)])

class ExtendedAckleyModel(Model):

    def __init__(
            self,
            a=20,
            b=0.2,
            c=2 * np.pi,
            e=1,
            d=10,
            
        ):
        self.a = a
        self.b = b
        self.d = d

        if isinstance(c, (int, float)):
            self.c = np.array([c] * d)
        else:
            assert len(c) == d, "Length of c must match the dimension d."
            self.c = np.array(c)

        if isinstance(e, (int, float)):
            self.e = np.array([e] * d)
        else:
            assert len(e) == d, "Length of e must match the dimension d."
            self.e = np.array(e)

    def __str__(self):
        return f"ExtendedAckleyModel(a={self.a}, b={self.b}, c={self.c}, e={self.e})"
        
    def evaluate(self, x):
        """
        Evaluate the Ackley function at a given point x.
        
        :param x: Input array representing the point at which to evaluate the function.
        :type x: np.ndarray
        :return: The value of the Ackley function at the given point x.
        :rtype: float
        """
        d = len(x)
        assert d == self.d, "Input dimension must match the model dimension."
        
        sum_sq = np.sum(self.e * x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        term1 = -self.a * np.exp(-self.b * sum_sq / d)
        term2 = -np.exp(sum_cos / d)
        return np.array([term1 + term2 + self.a + np.exp(1)])

class EggboxModel(Model):
    """
    Eggbox function model for synthetic testing.
    """
    def __init__(
            self,
            a=1,
            f=1
        ):
        """
        Initialize the Eggbox function model.
        
        :param a: Meta parameter - global output offset.
        :param f: Meta parameter - frequency scalar.
        """
        self.a = a
        self.f = f

    def __str__(self):
        print_a = "{:.2f}".format(self.a)
        print_f = "{:.2f}".format(self.f)
        return f"EggboxModel(a={print_a}, f={print_f})"

    def evaluate(self, x: np.ndarray) -> np.array:
        """
        Evaluate the Eggbox function at a given point x.
        
        :param x: Input array representing the point at which to evaluate the function.
        :type x: np.ndarray
        :return: The value of the Eggbox function at the given point x.
        :rtype: float
        """
        d = len(x)
        assert d > 0, "Input dimension must be greater than 0."
        sum_cos = np.sum(np.cos(self.f * x))
        return np.array([self.a * (d - sum_cos)])

class TemperedModel(Model):

    def __init__(self, model: Model, observed: float, scalar: float = 1.0):
        self.model = model
        self.observed = observed
        self.scalar = scalar

    def __str__(self):
        return self.model.__str__()


    def evaluate(self, x):
        # noise is internally a normal distribution
        noise_std = x[0]
        model_output = self.model.evaluate(x[1:])
        loglike = tda.GaussianLogLike(np.array([self.observed]), np.reshape(noise_std**2, (1, -1))).loglike(model_output)
        normal_factor = -np.log(noise_std * np.sqrt(2 * np.pi))
        return np.array([loglike + normal_factor]) * self.scalar

class TransformedModel(Model):

    def __init__(self, model: Model, transform: np.ndarray):
        self.model = model
        self.transform = transform

    def __str__(self):
        return self.model.__str__()
    
    def evaluate(self, x):
        return self.model.evaluate(self.transform @ x)

class DummyLoglike():

    def loglike(self, model_output):
        return model_output.item()

class ScalableGaussianLogLike():

    def __init__(self, observed, noise_cov, scalar=1.0):
        self.loglike_object = tda.GaussianLogLike(observed, noise_cov)
        self.scalar = scalar

    def loglike(self, x):
        return self.loglike_object.loglike(x) * self.scalar

def rotate_posterior_samples(idata: az.InferenceData, rot: np.ndarray) -> az.InferenceData:

    da = idata.posterior.to_array()
    non_sample_dims = [d for d in da.dims if d not in ("chain", "draw")]
    da = da.stack(sample=("chain", "draw")).stack(features=non_sample_dims)
    rotated = xr.apply_ufunc(
        lambda x: x @ rot.T,
        da.transpose("sample", "features"),
        input_core_dims=[["features"]],
        output_core_dims=[["features"]],
        vectorize=True
    )

    rotated = rotated.transpose("features", "sample")

    # Restore original structure
    rotated = rotated.unstack("features")
    rotated = rotated.unstack("sample")

    # Convert back to Dataset
    rotated_ds = rotated.to_dataset(dim="variable")

    idata_new = idata.copy()
    idata_new.posterior = rotated_ds

    return idata_new

def random_rotation_matrix(d, seed=None):
    """
    Generate a random rotation matrix of dimension d x d.
    
    :param d: Dimension of the rotation matrix.
    :param seed: Optional random seed for reproducibility.
    :return: A random rotation matrix of shape (d, d).
    :rtype: np.ndarray
    """
    
    rng = np.random.default_rng(seed)
    
    # Generate a random matrix
    random_matrix = rng.normal(size=(d, d))
    
    # Perform QR decomposition to obtain an orthogonal matrix
    q, r = np.linalg.qr(random_matrix)
    
    # Ensure a proper rotation (determinant = 1)
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    
    return q

if __name__ == "__main__":
    model_list = {}

    # prior setup
    d = 10
    #prior_mean = 3.5
    #prior_std = 3.5

    #d = 6
    prior_mean = 0.0
    prior_std = 2

    prior_noise_mean = 0.6
    prior_noise_std = 0.2

    #scale = np.array([10, 10, 10, 1, 1, 1, 1, 0.1, 0.1, 0.1])
    scale = np.array([1] * d)
    #rot = random_rotation_matrix(d, seed=1)
    rot = np.eye(d)
    
    assert np.allclose(rot.T @ rot, np.eye(rot.shape[0])), "Rotation matrix is not orthogonal."
    assert np.allclose(np.linalg.norm(rot, axis=0), 1), "Rotation matrix columns are not unit vectors"

    transform = rot @ np.diag(scale)

    #prior_mean = [prior_noise_mean] + [prior_mean] * (d - 1)
    #prior_cov = np.eye(d) * (np.array([prior_noise_std] + [prior_std] * (d-1)) ** 2)
    prior_mean = [prior_mean] * d
    prior_cov = (np.eye(d) * (np.array([prior_std] * d) ** 2))

    prior = multivariate_normal(np.array(prior_mean), prior_cov)

    # likelihood setup
    observed = 0
    noise_std = 1
    likelihood = ScalableGaussianLogLike(np.array([observed]), np.reshape(noise_std**2, (1, -1)), scalar=1.0)
    #likelihood = tda.AdaptiveGaussianLogLike(np.array([observed]), np.reshape(noise_std**2, (1, -1)))
    # tempered model has likelihood already built in, so we just pass that value along
    #likelihood = DummyLoglike()

    # Ackley model setup
    a = 6
    #c = 2 * np.pi
    c = 2 * np.pi * np.array([10, 10, 10, 1, 1, 1, 1, 0.1, 0.1, 0.1])
    #b = 3 * c
    b = 0.5
    e = 0.2
    #e = [10, 10, 10, 1, 1, 1, 1, 0.1, 0.1, 0.1]
    scalar = 4.0

    #ackley = AckleyModel(a, b, c, d)
    ackley = ExtendedAckleyModel(a, b, c, e, d=d)
    ackley = TransformedModel(ackley, transform)
    #ackley = TemperedModel(ackley, observed, scalar)
    model_list["Ackley"] = ackley

    # Eggbox model setup
    f = 2 * np.pi * 1
    a = 3
    anisotropic = False
    eggbox = EggboxModel(a, f)
    eggbox = TransformedModel(eggbox, transform)
    #eggbox = TemperedModel(eggbox, observed, scalar)
    #model_list["Eggbox"] = eggbox

    # proposal setup
    M0 = 1000
    DELTA = 1
    B = 0.1
    B_STAR = 1e-5
    Z_METHOD = 'lhs'
    NCR = 3
    ADAPTIVE = True
    ADAPTIVITY_PERIOD = 100
    GAMMA = 1.01
    SYNC_RATE = 500
    STUCK_CHECKING_START = 1e6
    STUCK_CHECKING_PERIOD = 1e6

    # sampler setup
    N_SAMPLES = 10000
    N_CHAINS = 6
    SEQUENTIAL = False


    proposal = tda.DREAM(
        M0 // N_CHAINS,
        DELTA,
        B,
        B_STAR,
        Z_METHOD,
        NCR,
        ADAPTIVE,
        GAMMA,
        ADAPTIVITY_PERIOD,
        0,
        SYNC_RATE,
        1,
        STUCK_CHECKING_START,
        STUCK_CHECKING_PERIOD
    )

    # proposal = tda.DREAMZ(
    #     M0,
    #     DELTA,
    #     B,
    #     B_STAR,
    #     Z_METHOD,
    #     NCR,
    #     ADAPTIVE,
    #     GAMMA,
    #     ADAPTIVITY_PERIOD)

    proposal = tda.GaussianRandomWalk(np.eye(d), scaling=4, adaptive=ADAPTIVE, period=ADAPTIVITY_PERIOD)

    # iterate over models
    for name, model in model_list.items():

        # construct posterior
        posterior = tda.Posterior(prior, likelihood, model.evaluate)

        samples = tda.sample(posterior, proposal, N_SAMPLES, N_CHAINS, force_sequential=SEQUENTIAL)
        idata = tda.to_inference_data(samples)

        idata_rotated = rotate_posterior_samples(idata, rot.T)

        print(az.summary(idata))
        save_idata_to_file(idata, f"test.idata")

        figs = []

        title_fontsize = 24

        title = f"{str(model)}, dim {d}, noise {noise_std}, {N_SAMPLES} samples, {N_CHAINS} chains, prior mean {prior_mean[0]}, prior std {prior_std}"
        pair_fig = az.plot_pair(idata, kind="kde")[0, 0].figure
        pair_fig.suptitle(title, wrap=True, fontsize=title_fontsize)
        figs.append(pair_fig)

        pair_fig_rotated = az.plot_pair(idata_rotated, kind="kde")[0, 0].figure
        pair_fig_rotated.suptitle(f"{title} (Rotated)", wrap=True, fontsize=title_fontsize)
        figs.append(pair_fig_rotated)

        trace_fig = plot_trace_modified(idata)[0, 0].figure
        trace_fig.suptitle("trace_plot", wrap=True, fontsize=title_fontsize)
        figs.append(trace_fig)

        trace_fig_rotated = plot_trace_modified(idata_rotated)[0, 0].figure
        trace_fig_rotated.suptitle("trace_plot (Rotated)", wrap=True, fontsize=title_fontsize)
        figs.append(trace_fig_rotated)

        prog_plots = plot_likelihood(idata, idata)
        figs.append(prog_plots[0].figure)
        figs.append(prog_plots[2].figure)

        save_plots_pdf_pages(f"{name}_results.pdf", figs)
