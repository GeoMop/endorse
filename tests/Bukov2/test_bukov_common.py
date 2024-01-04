import numpy as np
import matplotlib.pyplot as plt
from endorse.Bukov2 import bukov_common as bcommon

def test_soft_pressure():
    pressure = np.linspace(-30, 60)
    lim_p = bcommon.soft_lim_pressure(pressure)
    plt.plot(pressure, lim_p)
    plt.plot(pressure, np.maximum(0.13, pressure + 10))
    plt.show()