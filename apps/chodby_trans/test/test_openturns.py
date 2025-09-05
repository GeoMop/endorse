import math
import numpy as np
import openturns as ot
from typing import List, Tuple, Dict, Any
from chodby_trans import sa

def sobol_g_model(a_vec):
    d = len(a_vec)
    symbols = [f"X{i+1}" for i in range(d)]
    factors = [f"(abs(4*{s} - 2) + {a_vec[i]})/{1.0 + a_vec[i]}" for i, s in enumerate(symbols)]
    return ot.SymbolicFunction(symbols, [" * ".join(factors)])


def make_sobol_result(ot_result, param_names) -> sa.SobolResult:
    return sa.SobolResult(
        param_names,
        list(ot_result.getFirstOrderIndices()),
        list(ot_result.getFirstOrderIndicesInterval().getLowerBound()),
        list(ot_result.getFirstOrderIndicesInterval().getUpperBound()),
        list(ot_result.getTotalOrderIndices())
    )

def run_case(exp, method, model, second_order=False):
    exp_name, exp = exp
    sobol_exp = ot.SobolIndicesExperiment(exp, second_order)
    inputs_design = sobol_exp.generate()
    outputs = np.array(list(map(model, np.array(inputs_design))))
    output_design = ot.Sample(outputs)
    result = method(inputs_design, output_design, exp.getSize())

    name = f"{method.__name__.removesuffix("SensitivityAlgorithm")}-{exp_name}"
    result = sa.get_sensitive_params(
        make_sobol_result(result, inputs_design.getDescription()),
        threshold=0.95)
    return name, result

# -----------------------------
# Example usage
# -----------------------------
def test_open_turns():
    ot.RandomGenerator.SetSeed(42)
    size = 2**12
    n_params = 8
    distr = ot.JointDistribution([ot.Uniform(0.0, 1.0)] * n_params)
    second_order = False

    seq = ot.SobolSequence(n_params)
    experiments = dict(
        mc=ot.MonteCarloExperiment(distr, size),
        sobol=ot.LowDiscrepancyExperiment(seq, distr, size, False),
        lhs=ot.LHSExperiment(distr, size, False, False)
    )


    sa_methods = [ot.JansenSensitivityAlgorithm, ot.MartinezSensitivityAlgorithm,
                  ot.SaltelliSensitivityAlgorithm, ot.MauntzKucherenkoSensitivityAlgorithm]

    a_vec = [0.0, 0.5, 1.0, 3.0, 9.0, 20.0, 50.0, 50.0, 99.0, 99.0]
    model = sobol_g_model(a_vec[:n_params])

    cases = dict([
            run_case(exp, method, model)
            for exp in experiments.items()
        for method in sa_methods
        ])
    sa.sobol_plot(
        list(cases.values()),
        x=list(cases.keys()),
        show=True)
    #
    # compare = run_sobol_compare(
    #     methods=[
    #         ("Jansen-QMC",    ot.JansenSensitivityAlgorithm,    "qmc_sobol"),
    #         ("Martinez-MC",   ot.MartinezSensitivityAlgorithm,  "mc"),
    #         ("Saltelli-QMC",  ot.SaltelliSensitivityAlgorithm,  "qmc_sobol"),
    #         ("MauntzK-QMC",   ot.MauntzKucherenkoSensitivityAlgorithm, "qmc_sobol"),
    #     ],
    #     n=2**12,
    #     conf_level=0.90,
    #     bootstrap_size=256,
    #     stacked_threshold=0.8,
    #     base_width=0.6,
    #     max_extra_ratio=1.5,
    #     x_labels=None,     # will default to method labels
    #     fname=None,        # e.g., "compare_methods.pdf"
    #     show_plot=True
    # )
