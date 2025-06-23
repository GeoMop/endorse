# Test of WPT model behaviour for various model parameters.
from wpt_model import *

# create an array of size N having evenly distributed values
def make_het_array(N: int, values: float) -> np.array:
    a = values[0]*np.ones(N)
    n_values = len(values)
    for i,v in enumerate(values):
        a[i*N//n_values:(i+1)*N//n_values] = v
    return a

def format_exponential(lst, precision=0):
    return "[" + ", ".join(f"{num:.{precision}e}" for num in lst) + "]"

if __name__ == '__main__':
    # Geometry and time-stepping parameters.
    r_b = 0.2  # Borehole radius [m]
    R = 2.0  # Outer domain radius [m]
    N = 50  # Number of finite elements (=> N+1 nodes)
    dt = 24*60*60  # Time step [s]
    T_final = dt * 90  # Total simulation time (e.g., 3 days) [s]
    p_b0 = 1000*1000  # Borehole pressure (node 0) [Pa]
    p_far = 300*1000  # Far-field Dirichlet pressure (last node) [Pa]

    # Instantiate the solver.
    solver = PoroElasticSolver(r_b, R, N, dt, T_final, p_b0)

    # Homogeneous rock and fluid parameters.
    biot = 0.3
    phi = 0.05 # Porosity (e.g., 2%)
    E_values = [30e9, 30e6]  # homogeneous / heterogeneous values of Young's modulus
    nu_array = 0.25 * np.ones(N)   # Poisson's ratio
    #C_b = np.pi * r_b ** 2 * (c_f + (2 * (1 - nu) * (1 - nu ** 2)) / E)
    k_values = [1e-16, 1e-15] # homogeneous / heterogeneous values of conductivity

    # Run simulation ans save results.
    conductivity_array = make_het_array(N, [k_values[0]])
    E_array = make_het_array(N, [E_values[0]])
    time_vec, full_pressure_history, p_b_history \
        = solver.simulate(biot, phi, E_array, nu_array, p_far, conductivity_array)
    results = [{'p_b_history': p_b_history, 'full_pressure_history': full_pressure_history,
                'label': f'homogeneous K={k_values[0]} E={E_values[0]:.0e}'}]

    # Heterogeneous K:
    conductivity_array = make_het_array(N, k_values)
    E_array            = make_het_array(N, [E_values[0]])
    _, full_pressure_history, p_b_history \
        = solver.simulate(biot, phi, E_array, nu_array, p_far, conductivity_array)
    results.append({'p_b_history': p_b_history, 'full_pressure_history': full_pressure_history,
                    'label': f'heterogeneous K{k_values}'})

    # Heterogeneous E:
    conductivity_array = make_het_array(N, [k_values[0]])
    E_array            = make_het_array(N, E_values)
    _, full_pressure_history, p_b_history \
        = solver.simulate(biot, phi, E_array, nu_array, p_far, conductivity_array)
    results.append({'p_b_history': p_b_history, 'full_pressure_history': full_pressure_history,
                    'label': f'heterogeneous E{format_exponential(E_values)}'})

    # Heterogeneous K,E:
    conductivity_array = make_het_array(N, k_values)
    E_array            = make_het_array(N, E_values)
    _, full_pressure_history, p_b_history \
        = solver.simulate(biot, phi, E_array, nu_array, p_far, conductivity_array)
    results.append({'p_b_history': p_b_history, 'full_pressure_history': full_pressure_history,
                    'label': 'heterogeneous K,E'})


    # Plot results.
    solver.plot_results(results)
    # solver.plot_full_history()
