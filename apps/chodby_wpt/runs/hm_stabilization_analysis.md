# Source investigation of HM stabilization

Source checkout: `/home/jb/workspace/flow123d`, commit `8ff227a68`, matching
the version reported by the executable in these experiments. No solver source
was modified. All new cases use the identical S02 mesh and rendered input,
contact enabled, constant fracture fields, alpha=0.2 unless stated otherwise,
and the accepted strict/default inner solver settings.

## Implemented splitting

`src/coupling/hm_iterative.cc:42` implements

    beta = iteration_parameter * alpha^2 * rho*g / (2*Kd)
    Kd = lambda + 2*mu/3

`fn_fluid_source` (line 58) implements

    extra_source = [beta*(h_iter - h_previous_time)
                    - alpha*(div_u_iter - div_u_previous_time)] / dt

`update_flow_fields` passes beta as extra storativity. Darcy assembly
(`src/flow/assembly_lmh.hh:425`) multiplies BOTH extra source and storage
by element measure, cross-section and the lumping weight. At convergence the
artificial beta terms cancel. This is fixed-stress stabilization, not pressure
under-relaxation and not additional physical storage.

With the present material values and rho*g=9810, default beta is
4.905e-9 /m in rock (Kd=40 GPa), 7.848e-9 /m in packers (Kd=25 GPa), and
2.943e-5 /m in water/fractures (Kd=6.667 MPa). Compared with physical storage,
beta/S is about 0.0175, 0.0280 and 6.54, respectively. Thus the local
stabilization already differs dramatically between rock and soft domains.

For a linear, consistently coupled discretization, the pressure-error map has
the schematic form T=(H+L)^(-1)(L-C), where H contains physical storage and
dt times the Darcy operator, L is artificial storage, and C is the mechanical
pressure-to-volume response. Increasing storage or conductivity strengthens H;
changing iteration_parameter changes L. These are distinct mechanisms.

Reference: Storvik et al., *On the optimization of the fixed-stress splitting
for Biot's equations*, https://arxiv.org/abs/1811.06242. The optimum depends
on flow properties and discretization; the source's description of multiplier
1 as theoretically optimal is not a universal guarantee for this geometry.

## Concrete input defect

The S01-S07 inputs omit `bc_type: dirichlet` on `.pocket_water`. Their logs
explicitly list this boundary under default `bc_type: none`. Consequently,
`bc_pressure: 300` alone did not impose injection pressure. Earlier statements
attributing their instability to a sudden 300 m injection load were incorrect.
An explicit-Dirichlet experiment is included below; it still diverges, so this
defect is not the sole cause of the observed iteration instability.

## Mixed-dimensional coupling to investigate

`src/mechanics/assembly_elasticity.hh:483` uses `potential_load(p_high)` in
the interface RHS, weighted by fracture_sigma and the higher-dimensional
cross-section. In contrast, the output-divergence assembly (line 642) adds
normal displacement divided by fracture aperture to the lower-dimensional
divergence field. HM feeds that field back with the lower-dimensional alpha.
The bulk mechanics pressure load also has its own term (line 344).

This is a concrete place to check whether the two discrete coupling operators
are adjoints under the actual mass weights. Standard continuum fixed-stress
bounds do not establish that property here. This is a hypothesis, not a proven
implementation defect; proving it requires comparing the assembled operators
or testing a carefully derived consistent interface formulation.

The explicit-pocket case still diverges. Raising rock conductivity to the
fracture value reduces the differences greatly, but the absolute difference
eventually grows slowly: a falling relative norm alone is misleading because
the pressure norm in its denominator can grow too. Turning off Biot only in
fractures/water, while keeping alpha=0.2 in rock/packers, also fails. Thus the
feedback cannot be localized solely to soft-domain Biot terms by these tests.

Inspection of iteration updates found that Darcy matrix/RHS and mechanics
output-divergence vectors are zeroed before reassembly, and previous-time
pressure is accepted only after HM convergence. These checks did not reveal
a simple accumulation or premature time-advance bug. They do not exclude a
field-cache or discretization defect. The present conclusion is a narrowed
diagnosis, not a verified unique root cause or a stable production parameter set.

The experiments also retain initial stress and the highly compliant chamber.
Stiffening either fractures alone or water alone does not remove divergence.
Larger beta (2 or 10) slows but does not remove growth. Therefore a claim that
simply increasing stabilization fixes this case is unsupported.

Even setting every material to E=60 GPa and nu=0.25 (S17) fails:
at HM iteration 96 the absolute difference is 4.77794e6 and relative difference
0.145283. This rules out mechanical material contrast alone as a necessary
condition for the observed instability, though hydraulic contrasts remain.

## Protocol and limits

S08 onward use end_time=1 to isolate the first failing time step. A timeout
of 180 seconds executes INSIDE the container; exit 124 is an experimental
timeout, not a Flow123d convergence failure. No full-time stability claim can
be made from a first-step test. `hm_log_summary.py` extracts first-ten norms,
time steps, the final sequence and reported solver errors from the raw logs.
Exact scenario changes and results are in `hm_stability_experiments.md`.

## Homogeneous ultimate controls (S18-S29)

The hydraulic control removes all contrasts: every region has conductivity=1
and storativity=1. It also uses an explicit pocket Dirichlet head equal to the
initial head plus 10 m, constant fracture apertures, contact, common nu=0.25,
and zero initial stress.

With a common E=1 kPa (the mechanics integration-test scale), alpha=1
diverges, alpha=0.2 decreases impractically slowly, and alpha=0.02 stalls near
abs=0.0671. Changing `iteration_parameter` from 0 through 0.1, 1, 10 and 100
does not yield convergence. Multiplier 100 initially contracts but bottoms at
iteration 31 (abs=1.28163, rel=0.00233727), then grows. Tightening mechanics
`r_tol` from 1e-8 to 1e-12 leaves the alpha=0.02 trace identical to printed
precision, ruling out the mechanics linear tolerance as that plateau's cause.

With a common E=60 GPa, the same cases are accepted in three HM iterations for
alpha=1, 0.2 and 0.02. Their third absolute updates are 1.17421e-5,
4.69684e-7 and 4.69609e-9 respectively, i.e. they scale as alpha squared.

A stiffness bracket at alpha=1 changes common E from 10 MPa to 100 MPa and
1 GPa. After the initial transient, the absolute update plateaus at about
0.01678, 0.001678 and 0.0001678: exactly proportional to 1/E and persistent
for 20-34 observed iterations. Thus the E=60 GPa result is only a
tolerance-masked stable control: extrapolation puts its nonzero plateau below
the HM `r_tol`, so the algorithm accepts it rather than demonstrating true
fixed-point contraction.

These controls show that the problem can occur with no hydraulic or mechanical
material contrast. The persistent increment scales as alpha squared divided by
E and is unchanged by a tighter mechanics solve. That is strong numerical
evidence of a non-cancelling HM coupling contribution (or an eigenvalue-one
coupling mode), rather than ordinary slow convergence caused by material
contrast. The stabilization multiplier changes the transient path but does not
remove this mode; multiplier 100 eventually turns from contraction to growth.

The direct mesh control (S33) removes the node-copying pocket split, removes
both pocket boundary conditions, and applies a uniform water source density of
1e-3 over the full water volume. It converges in three HM iterations with
differences nearly identical to the pocketed S27 case. Gmsh reports no
ill-shaped tetrahedra, and the no-split mesh contains no `.pocket_packer_near`
region or copied interface nodes. This rules out the pocket mesh construction
as the primary cause and shifts the second hypothesis—the Flow123d branch or
its mixed-dimensional HM coupling—to the leading suspect.
