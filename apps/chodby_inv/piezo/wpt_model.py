import numpy as np
import matplotlib.pyplot as plt


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
        """
        self.r_b = r_b
        self.R = R
        self.N = N  # number of finite elements
        self.dt = dt
        self.T_final = T_final
        self.n_dofs = N + 1  # full number of nodes
        self.r = np.linspace(r_b, R, self.n_dofs)
        self.Nt = int(np.ceil(T_final / dt))
                # Define water parameters.
        self.c_f = 4.5e-10  # Water compressibility [Pa⁻¹]
        self.rho_f = 1e3    # Water density [kg/m³]
        self.g = 9.81       # m/s^2

    def interior_matrices(self, s_array, k_array):
        """
        Allocate full (n_dofs×n_dofs) mass and stiffness matrices and fill only the interior
        block (indices 1:n_dofs, 1:n_dofs) using P1 finite elements in cylindrical coordinates.

        Parameters:
          s_array: Piecewise constant storativity.
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
            s_val = s_array[i]
            M_local = s_val * r_mid * h_elem / 6.0 * np.array([[2, 1],
                                                           [1, 2]])
            # Element stiffness matrix.
            k_val = k_array[i]
            K_local = k_val * r_mid / h_elem * np.array([[1, -1],
                                                         [-1, 1]])
            # Map local indices to global indices: local {0,1} → global {i, i+1}.
            M[i:i + 2, i:i + 2] += M_local
            K[i:i + 2, i:i + 2] += K_local
        return M, K

    def assembly_matrices(self, s_array, k_array, C_b, dt):
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
        M_global, K_global = self.interior_matrices(s_array, k_array)
        # For the borehole node (node 0): override its mass row.
        M_global[0, :] = 0.0
        M_global[0, 0] = C_b
        # Compute alpha using the first element (connecting node 0 and node 1).
        h0 = self.r[1] - self.r[0]
        alpha = (2 * np.pi * self.r[0] * k_array[0]) / h0
        # Apply symmetric coupling for the flux between node 0 and node 1.
        K_global[0:2, 0:2] += alpha * np.array([[1, -1],
                                                [-1, 1]])
        return M_global, K_global

    def build_global_system(self, s_array, k_array, C_b, dt):
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
        M_global, K_global = self.assembly_matrices(s_array, k_array, C_b, dt)
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

    def estimate_complience(self, biot, phi, E, nu):
        """
        # Borehole compliance.

        Parameters:
          biot : Biot coefficient (dimensionless).
          phi  : Porosity (dimensionless).
          E    : Young's modulus [Pa].
          nu   : Poisson's ratio.

        Returns:
          c_s  : Rock compressibility [Pa⁻¹].
        """
        C_b = np.pi * self.r_b ** 2 * (self.c_f + (2 * (1 - nu) * (1 - nu ** 2)) / E)
        return C_b

    def estiamte_storativity(self, biot, phi, E, nu):
        """
        Estimate the storativity of the rock using Biot's theory.

        Parameters:
          biot : Biot coefficient (dimensionless).
          phi  : Porosity (dimensionless).
          E    : Young's modulus [Pa].
          nu   : Poisson's ratio.

        Returns:
          S    : Storativity [Pa⁻¹].
        """
        c_s = 3.0 * (1 - 2 * nu) / E
        S = phi * self.c_f + (biot - phi) * c_s
        return S

    def simulate(self, biot, phi, E, nu, p_init,  p_far, k_array, C_b_rel=1.0):
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


        # Compute rock compressibility.
        c_s = 3.0 * (1 - 2 * nu) / E
        # Storage coefficient (Is this correct ???)
        S = phi * self.c_f + (biot - phi) * c_s

        # currently just homogeneous storativity
        s_array = S * np.ones_like(k_array)
        # Borehole compliance.
        C_b = np.pi * self.r_b ** 2 * (self.c_f + (2 * (1 - nu) * (1 - nu ** 2)) / E)
        C_b = C_b * C_b_rel

        k_array = k_array / (self.rho_f * self.g)
        #m/s * m3/N * Pa/m = m/s * m3/N * N/m3 = m/s

        self.p_far = p_far

        # Build the global system.
        A_full, M_lumped, b_dirichlet = self.build_global_system(s_array, k_array, C_b, self.dt)

        n = self.n_dofs
        # Initialize the pressure field: node 0 = p_b0, interior nodes = p_far.
        p = np.ones(n) * self.p_far
        p[0] = p_init

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

    def plot_results(self):
        """ Plot borehole pressure history and final radial pressure distribution. """
        plt.figure(figsize=(10, 6))
        plt.plot(self.time_vec / 3600, self.p_b_history, label='Borehole Pressure p(0)')
        plt.xlabel('Time (hours)')
        plt.ylabel('Pressure (Pa)')
        plt.title('Borehole Pressure vs Time')
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.r, self.full_pressure_history[-1, :], marker='o')
        plt.xlabel('Radial Distance r (m)')
        plt.ylabel('Pressure (Pa)')
        plt.title('Pressure Distribution at Final Time')
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
    E = 30e9  # Young's modulus (e.g., 50 GPa)
    nu = 0.25  # Poisson's ratio
    C_b = np.pi * r_b ** 2 * (c_f + (2 * (1 - nu) * (1 - nu ** 2)) / E)

    # Hydraulic conductivity: piecewise constant (array of length N).
    conductivity_array = 1e-16 * np.ones(N)
    # Optionally, introduce heterogeneity:
    # conductivity_array[N//3:2*N//3] = 2e-12

    # Run simulation.
    time_vec, full_pressure_history, p_b_history \
        = solver.simulate(biot, phi, E, nu, p_far, conductivity_array)

    # Plot results.
    solver.plot_results()
    solver.plot_full_history()
