import numpy as np
import arviz as az
from ray import logger
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


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
            anisotropic=False,
            scale=None
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
        self.anisotropic = anisotropic
        if self.anisotropic:
            assert scale is not None, "Scale must be provided for anisotropic Ackley model."
            self.scale = scale

    def __str__(self):
        print_a = "{:.2f}".format(self.a)
        print_b = "{:.2f}".format(self.b)
        print_c = "{:.2f}".format(self.c)
        return f"AckleyModel(a={print_a}, b={print_b}, c={print_c}, anisotropic={self.anisotropic}, scale={self.scale if self.anisotropic else None})"

    def evaluate(self, x: np.ndarray) -> np.array:
        """
        Evaluate the Ackley function at a given point x.
        
        :param x: Input array representing the point at which to evaluate the function.
        :type x: np.ndarray
        :return: The value of the Ackley function at the given point x.
        :rtype: float
        """
        d = len(x)
        assert d > 0, "Input dimension must be greater than 0."
        assert not self.anisotropic or len(self.scale) == d, "Scale length must match input dimension for anisotropic Ackley model."
        if self.anisotropic:
            x = x * self.scale
        
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return np.array([term1 + term2 + self.a + np.exp(1)])

class EggboxModel(Model):
    """
    Eggbox function model for synthetic testing.
    """
    def __init__(
            self,
            a=1,
            f=1,
            anisotropic=False,
            scale=None
        ):
        """
        Initialize the Eggbox function model.
        
        :param a: Meta parameter - global output offset.
        :param f: Meta parameter - frequency scalar.
        """
        self.a = a
        self.f = f
        self.anisotropic = anisotropic
        if self.anisotropic:
            assert scale is not None, "Scale must be provided for anisotropic Eggbox model."
            self.scale = scale

    def __str__(self):
        print_a = "{:.2f}".format(self.a)
        print_f = "{:.2f}".format(self.f)
        return f"EggboxModel(a={print_a}, f={print_f}, anisotropic={self.anisotropic}, scale={self.scale if self.anisotropic else None})"

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
        assert not self.anisotropic or len(self.scale) == d, "Scale length must match input dimension for anisotropic Eggbox model."
        if self.anisotropic:
            x = x * self.scale
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

class DummyLoglike():

    def loglike(self, model_output):
        return model_output.item()

class ScalableGaussianLogLike():

    def __init__(self, observed, noise_cov, scalar=1.0):
        self.loglike_object = tda.GaussianLogLike(observed, noise_cov)
        self.scalar = scalar

    def loglike(self, x):
        return self.loglike_object.loglike(x) * self.scalar

if __name__ == "__main__":
    model_list = {}

    # prior setup
    d = 8
    prior_mean = 3.5
    prior_std = 3.5

    #d = 6
    #prior_mean = 0.0
    #prior_std = 1

    prior_noise_mean = 0.6
    prior_noise_std = 0.2

    prior_mean = [prior_noise_mean] + [prior_mean] * (d - 1)
    prior_cov = np.eye(d) * (np.array([prior_noise_std] + [prior_std] * (d-1)) ** 2)
    #prior_mean = [prior_mean] * d
    #prior_cov = np.eye(d) * (np.array([prior_std] * d) ** 2)
    prior = multivariate_normal(np.array(prior_mean), prior_cov)

    # likelihood setup
    observed = 0
    noise_std = 0.8
    #likelihood = ScalableGaussianLogLike(np.array([observed]), np.reshape(noise_std**2, (1, -1)), scalar=1.0)
    # tempered model has likelihood already built in, so we just pass that value along
    likelihood = DummyLoglike()

    # Ackley model setup
    a = 6
    c = 2 * np.pi
    #b = 3 * c
    b = 0.5
    anisotropic = False
    scalar = 4.0
    scale = np.array([10, 10, 10, 1, 1, 1, 1, 1, 1, 1])
    ackley = AckleyModel(a, b, c, anisotropic, scale)
    ackley = TemperedModel(ackley, observed, scalar)
    model_list["Ackley"] = ackley

    # Rastrigin model setup
    #rastrigin = RastriginModel(f=2, a=2)
    #model_list["Rastrigin"] = rastrigin

    # Eggbox model setup
    f = 2 * np.pi * 1
    a = 3
    anisotropic = False
    eggbox = EggboxModel(a, f, anisotropic, scale)
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
    SYNC_RATE = 1e6
    STUCK_CHECKING_START = 1e6
    STUCK_CHECKING_PERIOD = 1e6

    proposal = tda.DREAM(
        M0,
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

    # sampler setup
    N_SAMPLES = 2000
    N_CHAINS = 6
    SEQUENTIAL = False

    # iterate over models
    for name, model in model_list.items():

        # construct posterior
        posterior = tda.Posterior(prior, likelihood, model.evaluate)

        samples = tda.sample(posterior, proposal, N_SAMPLES, N_CHAINS, force_sequential=SEQUENTIAL)
        idata = tda.to_inference_data(samples)

        figs = []

        title = f"{str(model)}, dim {d}, noise {noise_std}, {N_SAMPLES} samples, {N_CHAINS} chains, prior mean {prior_mean[0]}, prior std {prior_std}"
        pair_fig = az.plot_pair(idata, kind="kde")[0, 0].figure
        pair_fig.suptitle(title, fontsize=32)
        figs.append(pair_fig)
        trace_fig = plot_trace_modified(idata)[0, 0].figure
        figs.append(trace_fig)
        prog_plots = plot_likelihood(idata, idata)
        figs.append(prog_plots[0].figure)
        figs.append(prog_plots[2].figure)

        save_plots_pdf_pages(f"{name}_results.pdf", figs)
