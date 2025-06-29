import numpy as np
import matplotlib.pyplot as plt

from typing import TypedDict, List

# Data type for passing results to plotting functions
class DataItem(TypedDict):
    p_b_results: np.ndarray
    full_pressure_history: np.ndarray
    label: str


class PoroElasticSolver:
    def __init__(self, r_b, R, N, dt, T_final, p_b0):
        """
        Stores geometry, time-stepping, and fixed pressures.

        Parameters:
          r_b   : Borehole radius [m].
          R     : Outer radius [m].
          N     : Number of finite elements (mesh will have N+1 nodes).
          dt    : Time step [s].
          T_final: Total simulation time [s].
          p_b0  : Fixed borehole pressure (node 0) [Pa].
          p_far : Fixed far-field (Dirichlet) pressure (last node) [Pa].
        """
        self.r_b = r_b
        self.R = R
        self.N = N  # number of finite elements
        self.dt = dt
        self.T_final = T_final
        self.n_dofs = N + 1  # full number of nodes
        self.r = np.linspace(r_b, R, self.n_dofs)
        self.Nt = int(np.ceil(T_final / dt))
        self.p_b0 = p_b0

    def interior_matrices(self, S_array, k_array):
        """
        Allocate full (n_dofs×n_dofs) mass and stiffness matrices and fill only the interior
        block (indices 1:n_dofs, 1:n_dofs) using P1 finite elements in cylindrical coordinates.

        Parameters:
          S_array: 1D array of length N with piecewise constant storage coefficient.
          k_array: 1D array of length N with piecewise constant hydraulic conductivity.

        Returns:
          M, K: Full arrays of shape (n_dofs, n_dofs) with only the interior block (rows/cols 1: end)
                filled.
        """
        n = self.n_dofs
        M = np.zeros((n, n))
        K = np.zeros((n, n))
        # Loop over interior elements only.
        # Interior nodes are 1, 2, ..., n-1 so elements connecting nodes i and i+1 for i=1,...,n-2.
        for i in range(1, self.N):  # i = 1,...,N-1 (since n = N+1)
            h_elem = self.r[i + 1] - self.r[i]
            r_mid = 0.5 * (self.r[i] + self.r[i + 1])
            # Consistent element mass matrix.
            S_val = S_array[i]
            M_local = S_val * r_mid * h_elem / 6.0 * np.array([[2, 1],
                                                           [1, 2]])
            # Element stiffness matrix.
            k_val = k_array[i]
            K_local = k_val * r_mid / h_elem * np.array([[1, -1],
                                                         [-1, 1]])
            # Map local indices to global indices: local {0,1} → global {i, i+1}.
            M[i:i + 2, i:i + 2] += M_local
            K[i:i + 2, i:i + 2] += K_local
        return M, K

    def assembly_matrices(self, S_array, k_array, C_b, dt):
        """
        Use interior_matrices to allocate full matrices and then insert borehole physics.

        For node 0 (borehole):
          - Set its mass row to zero except M[0,0] = C_b.
          - The interior block remains as assembled for nodes 1: n_dofs.

        Then add the symmetric coupling between node 0 and node 1.
        Compute alpha = 2π·r_b·k_array[0] / h₀, with h₀ = r[1]-r[0], and then do:

            K_global[0:2, 0:2] += alpha * np.array([[1, -1],
                                                     [-1,  1]])

        Returns:
          M_global, K_global: Full (n_dofs×n_dofs) matrices with borehole treatment applied.
        """
        n = self.n_dofs
        # Allocate full matrices and fill interior block.
        M_global, K_global = self.interior_matrices(S_array, k_array)
        # For the borehole node (node 0): override its mass row.
        M_global[0, :] = 0.0
        M_global[0, 0] = C_b[0]
        # Compute alpha using the first element (connecting node 0 and node 1).
        h0 = self.r[1] - self.r[0]
        alpha = (2 * np.pi * self.r[0] * k_array[0]) / h0
        # Apply symmetric coupling for the flux between node 0 and node 1.
        K_global[0:2, 0:2] += alpha * np.array([[1, -1],
                                                [-1, 1]])
        return M_global, K_global

    def build_global_system(self, S_array, k_array, C_b, dt):
        """
        Build the global system.

        Steps:
          1. Call assembly_matrices to get M_global and K_global.
          2. Lump the mass matrix: for each row take its sum and divide by dt to form M_lumped.
          3. Form A_full = diag(M_lumped) + K_global.
          4. Impose the Dirichlet condition at the last node (n-1) by transferring the contribution
             from the last column to a correction vector b_dirichlet with a negative sign, then
             setting the last row of A_full to the identity.

        Returns:
          A_full      : Global system matrix (n_dofs×n_dofs).
          M_lumped    : Lumped mass vector (divided by dt) (length n_dofs).
          b_dirichlet : Dirichlet correction vector (length n_dofs) (with a negative sign).
        """
        n = self.n_dofs
        M_global, K_global = self.assembly_matrices(S_array, k_array, C_b, dt)
        # Lump the mass matrix: sum each row divided by dt.
        M_lumped = np.sum(M_global, axis=1) / dt
        A_full = np.diag(M_lumped) + K_global

        # For Dirichlet condition at node n-1: the contribution from the last column is subtracted.
        # The correction vector is then b_dirichlet = -A_full[:, n-1] * p_far.
        b_dirichlet = -A_full[:, n - 1] * self.p_far
        # Replace the last row of A_full by the identity row.
        A_full[:, n-1] = 0.0
        A_full[n - 1, :] = 0.0
        A_full[n - 1, n - 1] = 1.0
        b_dirichlet[n - 1] = self.p_far
        return A_full, M_lumped, b_dirichlet

    def simulate(self, biot: float, phi: float, E: np.array, nu: np.array, p_far: float, k_array: np.array, C_b: np.array = None):
        """
        Run the fully implicit simulation.

        Parameters:
          phi     : Porosity (dimensionless).
          E       : Young's modulus [Pa].
          nu      : Poisson's ratio.
          k_array : 1D array (length N) with piecewise constant hydraulic conductivity.

        Physics:
          - Water compressibility: c_f = 4.5e-10 [Pa⁻¹].
          - Rock compressibility: c_s = 3*(1-2ν)/E.
          - Storage coefficient: S = phi*c_f + (1-phi)*c_s.
          - Borehole compliance:
                C_b = π*r_b² [ c_f + 2(1-ν)(1-ν²)/E ].

        The time step is computed as:
             p_new = solve( A_full, (M_lumped ⊙ p_current) + b_dirichlet ).

        Returns:
          time_vec             : 1D array of time [s].
          full_pressure_history: 2D array (Nt+1 × n_dofs) of pressures.
          p_b_history          : 1D array (length Nt+1) of borehole (node 0) pressures.
        """
        # Define water compressibility.
        c_f = 4.5e-10
        rho_f = 1e3
        g = 9.81  # m/s^2

        # Compute rock compressibility.
        c_s = 3.0 * (1 - 2 * nu) / E
        # Storage coefficient.
        S = phi * c_f + (biot - phi) * c_s
        # Borehole compliance.
        if C_b is None:
            C_b = np.pi * self.r_b ** 2 * (c_f + (2 * (1 - nu) * (1 - nu ** 2)) / E)

        # Save parameters.
        self.phi = phi;
        self.E = E;
        self.nu = nu;
        self.S = S;
        self.C_b = C_b;
        self.k_array = k_array / (rho_f * g)
        #m/s * m3/N * Pa/m = m/s * m3/N * N/m3 = m/s

        self.p_far = p_far

        # Build the global system.
        A_full, M_lumped, b_dirichlet = self.build_global_system(S, k_array, C_b, self.dt)

        n = self.n_dofs
        # Initialize the pressure field: node 0 = p_b0, interior nodes = p_far.
        p = np.ones(n) * self.p_far
        p[0] = self.p_b0

        self.time_vec = np.zeros(self.Nt + 1)
        self.p_b_history = np.zeros(self.Nt + 1)
        self.full_pressure_history = np.zeros((self.Nt + 1, n))
        self.full_pressure_history[0, :] = p.copy()
        self.p_b_history[0] = p[0]

        # Time-stepping loop.
        for t in range(self.Nt):
            # RHS: b = M_lumped ⊙ p + b_dirichlet.
            b = M_lumped * p + b_dirichlet
            p_new = np.linalg.solve(A_full, b)
            p = p_new.copy()
            p[n - 1] = self.p_far  # enforce Dirichlet
            self.time_vec[t + 1] = self.time_vec[t] + self.dt
            self.p_b_history[t + 1] = p[0]
            self.full_pressure_history[t + 1, :] = p.copy()
        return self.time_vec, self.full_pressure_history, self.p_b_history

    def plot_results(self, data: List[DataItem] = None):
        if data is None:
            data = [ { 'p_b_history': self.p_b_history,
                     'full_pressure_history': self.full_pressure_history,
                     'label': 'Borehole Pressure' } ]

        """ Plot borehole pressure history and final radial pressure distribution. """
        plt.figure(figsize=(10, 6))
        for d in data:
            plt.plot(self.time_vec / 3600, d['p_b_history'], label=d['label'])
        plt.xlabel('Time (hours)')
        plt.ylabel('Pressure (Pa)')
        plt.title('Borehole Pressure p(0) vs Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        for d in data:
            plt.plot(self.r, d['full_pressure_history'][-1, :], marker='o', label=d['label'])
        plt.xlabel('Radial Distance r (m)')
        plt.ylabel('Pressure (Pa)')
        plt.title('Pressure Distribution at Final Time')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_full_history(self):
        """ Plot a pseudocolor map of the full pressure field evolution. """
        plt.figure(figsize=(10, 6))
        T, R_mesh = np.meshgrid(self.time_vec / 3600, self.r)
        plt.pcolormesh(R_mesh, T, self.full_pressure_history.T, shading='auto')
        plt.xlabel('Radial Distance r (m)')
        plt.ylabel('Time (hours)')
        plt.title('Full Pressure Field History')
        cbar = plt.colorbar()
        cbar.set_label('Pressure (Pa)')
        plt.show()


# =====================
# Example Usage:
# =====================
if __name__ == '__main__':
    # Geometry and time-stepping parameters.
    r_b = 0.2  # Borehole radius [m]
    R = 2.0  # Outer domain radius [m]
    N = 50  # Number of finite elements (=> N+1 nodes)
    dt = 24*60*60  # Time step [s]
    T_final = dt * 90  # Total simulation time (e.g., 3 days) [s]
    p_b0 = 1000*1000  # Borehole pressure (node 0) [Pa]
    p_far = 300*1000  # Far-field Dirichlet pressure (last node) [Pa]
    c_f = 4.5e-10

    # Instantiate the solver.
    solver = PoroElasticSolver(r_b, R, N, dt, T_final, p_b0)

    # Rock and fluid parameters.
    biot = 0.3
    phi = 0.05  # Porosity (e.g., 2%)
    E_array = 30e9 * np.ones(N) # Young's modulus (e.g., 50 GPa)
    nu = 0.25  # Poisson's ratio
    C_b = np.pi * r_b ** 2 * (c_f + (2 * (1 - nu) * (1 - nu ** 2)) / E_array)

    # Hydraulic conductivity: piecewise constant (array of length N).
    conductivity_array = 1e-16 * np.ones(N)
    # Optionally, introduce heterogeneity:
    # conductivity_array[N//3:2*N//3] = 2e-12

    # Run simulation.
    time_vec, full_pressure_history, p_b_history \
        = solver.simulate(biot, phi, E_array, nu, p_far, conductivity_array)

    # Plot results.
    solver.plot_results()
    solver.plot_full_history()
