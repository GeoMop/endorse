import numpy as np
from scipy.stats import qmc


# ---------- Sobol-index computation for groups ----------

def sobol_indices_groups_from_matrices(fA, fB, fC_group_dict):
    """
    Compute S1 and ST for each *group* from already-evaluated model outputs.
    Uses Saltelli (S1) + Jansen (ST).
    """
    VY = np.var(np.concatenate([fA, fB]), ddof=1)
    S1 = {}
    ST = {}
    for g, fC in fC_group_dict.items():
        S1[g] = np.mean(fB * (fC - fA)) / VY          # S1
        ST[g] = 0.5 * np.mean((fA - fC) ** 2) / VY    # ST
    return S1, ST, VY


# ---------- samplers parameterised by m ----------

def mc_sampler(m, d, seed=None):
    """
    Monte Carlo sampler: returns N = 2**m points in [0,1]^d.
    """
    N = 2**m
    rng = np.random.default_rng(seed)
    return rng.random((N, d))


def sobol_sampler(m, d, seed=None):
    """
    Sobol QMC sampler: returns N = 2**m Sobol points in [0,1]^d
    using random_base2(m).
    """
    eng = qmc.Sobol(d=d, scramble=True, seed=seed)
    return eng.random_base2(m)


# ---------- matrix builders using m ----------

def build_AB_independent(m, d, sampler, seedA=None, seedB=None):
    """
    Build A, B with any sampler(m, d, seed) -> (N, d), where N = 2**m.
    """
    A = sampler(m, d, seedA)
    B = sampler(m, d, seedB)
    return A, B


def build_AB_hashed(m, d, seed_sampler, seedA=None, seedB=None):
    """
    'Hashed' case: each row is generated from a single seed u in [0,1].
    First column = seed, other columns = deterministic 'hash' of the seed.

    seed_sampler: function(m, 1, seed) -> (N, 1) MC or QMC points, N = 2**m.
    """
    seedsA = seed_sampler(m, 1, seedA).reshape(-1)
    seedsB = seed_sampler(m, 1, seedB).reshape(-1)
    N = seedsA.shape[0]

    A = np.zeros((N, d))
    B = np.zeros((N, d))
    A[:, 0] = seedsA
    B[:, 0] = seedsB

    def hash_like(x, j):
        # j = 1..d-1 to differentiate hashes
        return (np.sin((37.0 + 10 * j) * x + 0.123 * (j + 1)) * 1e4) % 1.0

    for j in range(1, d):
        A[:, j] = hash_like(seedsA, j)
        B[:, j] = hash_like(seedsB, j)

    return A, B


def build_C_group_matrices(A, B, groups):
    """
    Build C_G matrices by swapping *all columns in a group* from B into A.

    groups: dict[group_name] = list_of_column_indices (0-based).
    """
    C_group = {}
    for g, cols in groups.items():
        C = A.copy()
        C[:, cols] = B[:, cols]
        C_group[g] = C
    return C_group


# ---------- toy model & analysis ----------

# Example model: Y = X1 (only first parameter matters)
def model(X):
    return X[:, 0]


def analysis_with_groups(A, B, groups, label):
    C_group = build_C_group_matrices(A, B, groups)
    fA = model(A)
    fB = model(B)
    fC_group = {g: model(C_group[g]) for g in groups}
    S1, ST, VY = sobol_indices_groups_from_matrices(fA, fB, fC_group)

    print(label)
    print("  Var(Y) =", VY)
    for g in groups:
        print(f"  Group {g}: S1 = {S1[g]:.5f}, ST = {ST[g]:.5f}")
    print()


# ---------- run tests with m ----------

m = 16        # N = 2**m samples
d = 3

groups = {
    "param1": [0],
    "param2": [1],
    "param3": [2],
    "all":    [0, 1, 2],
}

# 1) MC independent
A_mc_ind, B_mc_ind = build_AB_independent(m, d, mc_sampler, seedA=1, seedB=2)
analysis_with_groups(A_mc_ind, B_mc_ind, groups, "MC independent")

# 2) MC hashed
A_mc_hash, B_mc_hash = build_AB_hashed(m, d, mc_sampler, seedA=3, seedB=4)
analysis_with_groups(A_mc_hash, B_mc_hash, groups, "MC hashed")

# 3) Sobol QMC independent
A_qmc_ind, B_qmc_ind = build_AB_independent(m, d, sobol_sampler, seedA=11, seedB=22)
analysis_with_groups(A_qmc_ind, B_qmc_ind, groups, "Sobol QMC independent")

# 4) Sobol QMC hashed (1D Sobol for seeds)
A_qmc_hash, B_qmc_hash = build_AB_hashed(m, d, sobol_sampler, seedA=33, seedB=44)
analysis_with_groups(A_qmc_hash, B_qmc_hash, groups, "Sobol QMC hashed")
