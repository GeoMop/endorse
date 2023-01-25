from endorse import common
import os
import numpy as np
from scipy import stats
import pytest

script_dir = script_dir = os.path.dirname(os.path.realpath(__file__))

def test_workdir():
    pass



def test_substitute_placeholders():
    pass


def test_check_conv_reasons():
    pass


def test_call_flow():
    pass


def test_sample_from_population():
    population = np.array([(i, i*i) for i in [1,2,3,4]])
    frequencies = [10, 3, 20, 4]
    i_samples = common.sample_from_population(10000, frequencies)
    samples = population[i_samples, ...]
    sampled_freq = 4 * [0]
    for i, ii in samples:
        sampled_freq[i-1] += 1
    chisq, pval = stats.chisquare(sampled_freq, np.array(frequencies) / np.sum(frequencies) * len(samples))
    print("\nChi square test pval: ", pval)
    assert pval > 0.05
