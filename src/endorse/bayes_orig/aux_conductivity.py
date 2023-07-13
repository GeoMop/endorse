
import os
import sys
import yaml

import matplotlib.pyplot as plt
import numpy as np
import scipy as scp

# rep_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(rep_dir)

def eps_func(sigma_vm, gamma_T, alpha, sigma_cT):

    eps_T = np.tanh(gamma_T*(sigma_vm - sigma_cT))
    eps = 0.5*((alpha-1)*(eps_T+1) + 2)
    return eps

def plot_eps_01():
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('sigma [Pa]')
    ax1.set_ylabel('eps [m]')

    sigma_c = 5.5e7

    # params
    gamma_T = np.array([0.5, 1, 2, 3, 5, 7, 9]) * 1e-7
    alpha = 1e7
    sigma_cT = 7e7

    x = np.arange(start=0, stop=1e8, step=1e6)

    for gt in gamma_T:
        y = eps_func(x, gt, alpha, sigma_cT)
        ax1.plot(x, y, color='black', label="gamma_T=" + str(gt), linestyle='solid')
        print("eps(sigma_c, gamma_t =", gt, ") = " +
              f'{eps_func(sigma_c, gt, alpha, sigma_cT):1.2e}',
              "    d/ds = " + f'{(alpha - 1) / 2 * gt:1.2f}')

    # ax1.set_yscale('log')
    colormap = plt.cm.gist_rainbow
    colors = [colormap(i) for i in np.linspace(0, 1, len(ax1.lines))]
    for i, j in enumerate(ax1.lines):
        j.set_color(colors[i])

    ax1.legend()

    ax1.axvline(x=sigma_c, ymin=0.0, ymax=1.0, color='k', linewidth=0.25)
    ax1.axvline(x=sigma_cT, ymin=0.0, ymax=1.0, color='k', linewidth=0.25)

    # ax1.tick_params(axis='y')
    # ax1.legend(ncol=3)

    fig.tight_layout()
    plt.show()

    # fig_file = os.path.join(self._config["work_dir"], "interp_data_TSX.pdf")
    # plt.savefig(fig_file)


def Sfunc(sigma,a,b):
    s = a*(sigma - b)
    sigmoid = np.exp(s) / (np.exp(s) + 1)
    return sigmoid

def eps_func_02(sigma,a,b):
    s = Sfunc(sigma, a, b)
    # return s
    # we want max 7 orders increase in conductivity:
    #       np.log10(kmax/k0) ~ 7
    #       np.log(kmax/k0) ~ 7/log10(e) ~ 7/0.43
    f = 7/(np.log10(np.e))
    return np.exp(f*s)

def plot_eps_02():
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('sigma [Pa]')
    ax1.set_ylabel('eps [m]')

    sigma_c = 5.5e7
    gamma = 3e-7

    # params
    # kf = np.log10(kmax)/np.log10(k0) ~ 7
    kf = 7
    a = 4*gamma / kf
    print(7/(np.log10(np.e)))
    # b = (np.exp(s) + 1) / np.exp(s)
    def nl_func(b):
        # np.log10(1.1) / 7 = np.log(1.1) / (7 / np.log10(np.e))
        return Sfunc(sigma_c, a, b) - np.log10(1.1)/7
    b = scp.optimize.fsolve(nl_func, 6e7)
    print(a,b)
    # def nl_func(b):
    #     return b**2 - 2
    # b = scp.optimize.fsolve(nl_func, 1.0)

    # a=2e-7
    # b=-7e7
    # print(a,b)

    # a = 2.5071296382476096e-07
    # b = 72748833.3695458
    # print(a,b)

    x = np.arange(start=0.75*sigma_c, stop=2.25*sigma_c, step=5e5)

    # for gt in gamma_T:
    y = eps_func_02(x, a, b)
    ax1.plot(x, y, color='black', label="eps_func_02", linestyle='solid')

    ax1.set_yscale('log')
    colormap = plt.cm.gist_rainbow
    colors = [colormap(i) for i in np.linspace(0, 1, len(ax1.lines))]
    for i, j in enumerate(ax1.lines):
        j.set_color(colors[i])

    ax1.legend()

    ax1.axvline(x=sigma_c, ymin=0.0, ymax=1.0, color='k', linewidth=0.25)
    ax1.text(sigma_c*1.01, 2, r"$[\sigma_c,1.1]$", rotation=0)
    ax1.axvline(x=b, ymin=0.0, ymax=1.0, color='k', linewidth=0.25)
    ax1.text(b * 1.01, 10**(kf/2), r"$[b,\sqrt{\frac{K_{max}}{K_0}}]$", rotation=0)

    ax1.axhline(y=1.1, xmin=0.0, xmax=1.0, color='k', linewidth=0.25)
    ax1.axhline(y=1e7, xmin=0.0, xmax=1.0, color='k', linewidth=0.25)

    # ax1.tick_params(axis='y')
    # ax1.legend(ncol=3)

    fig.tight_layout()
    # plt.show()

    # fig_file = os.path.join(self._config["work_dir"], "interp_data_TSX.pdf")
    plt.savefig("eps_func_02.pdf")


if __name__ == "__main__":
    # print(np.log10(np.e))
    # print(np.log10(1e7))
    #
    # print(np.log(1e7))
    # print(7/np.log10(np.e))
    #
    # print(np.log10(1.1)/7)
    # print(np.log(1.1) / (7/np.log10(np.e)))

    # plot_eps_01()
    plot_eps_02()

    # output_dir = None
    # len_argv = len(sys.argv)
    # assert len_argv > 1, "Specify output directory!"
    # if len_argv > 1:
    #     output_dir = sys.argv[1]


