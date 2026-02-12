import numpy as np
import arviz as az
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

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

    def evaluate(self, x: np.ndarray) -> np.array:
        """
        Evaluate the Ackley function at a given point x.
        
        :param x: Input array representing the point at which to evaluate the function.
        :type x: np.ndarray
        :return: The value of the Ackley function at the given point x.
        :rtype: float
        """
        d = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(self.c * x))
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / d))
        term2 = -np.exp(sum_cos / d)
        return np.array([term1 + term2 + self.a + np.exp(1)])

class RastriginModel(Model):
    """
    Rastrigin function model for synthetic testing.
    """
    def __init__(
            self,
            a=10,
            f=1
        ):
        """
        Initialize the Rastrigin function model.
        
        :param a: Meta parameter - global output scalar (?).
        :param f: Meta parameter - frequency scalar.
        """
        self.a = a
        self.f = f

    def evaluate(self, x: np.ndarray) -> np.array:
        """
        Evaluate the Rastrigin function at a given point x.
        
        :param x: Input array representing the point at which to evaluate the function.
        :type x: np.ndarray
        :return: The value of the Rastrigin function at the given point x.
        :rtype: float
        """
        d = len(x)
        sum_sq = np.sum(x**2)
        sum_cos = np.sum(np.cos(2 * np.pi * x))
        return np.array([self.a * d + sum_sq - self.a * sum_cos])

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

    def evaluate(self, x: np.ndarray) -> np.array:
        """
        Evaluate the Eggbox function at a given point x.
        
        :param x: Input array representing the point at which to evaluate the function.
        :type x: np.ndarray
        :return: The value of the Eggbox function at the given point x.
        :rtype: float
        """
        d = len(x)
        sum_cos = np.sum(np.cos(self.f * x))
        return np.array([self.a * (d - sum_cos)])


if __name__ == "__main__":

    ackley = AckleyModel(c=4*np.pi, b=0.15)
    rastrigin = RastriginModel(f=2, a=2)
    eggbox = EggboxModel(f=4, a=0.5)

    model_list = [ackley, rastrigin, eggbox]
    names_list = ["Ackley", "Rastrigin", "Eggbox"]

    # prior setup
    d = 4
    prior_mean = [3.5] * d
    prior_cov = np.eye(d) * 2
    prior = multivariate_normal(np.array(prior_mean), prior_cov)

    # likelihood setup
    observed = 0
    noise = 1
    likelihood = tda.GaussianLogLike(np.array([observed]), np.reshape(noise, (1, -1)))

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
    N_SAMPLES = 10000
    N_CHAINS = 6
    SEQUENTIAL = False

    # iterate over models
    for model, name in zip(model_list, names_list):

        # construct posterior
        posterior = tda.Posterior(prior, likelihood, model.evaluate)

        samples = tda.sample(posterior, proposal, N_SAMPLES, N_CHAINS, force_sequential=SEQUENTIAL)
        idata = tda.to_inference_data(samples)

        figs = []
        figs.append(az.plot_pair(idata, kind="kde")[0, 0].figure)
        figs.append(plot_trace_modified(idata)[0, 0].figure)
        figs.append(plot_likelihood(idata, idata)[0])
        save_plots_pdf_pages(f"{name}_results.pdf", figs)
