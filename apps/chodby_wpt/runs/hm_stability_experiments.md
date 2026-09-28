# HM stability experiment log

## S33 — no pocket split, uniform water influx

- Changes from S27: generated the mesh with `split_pocket=False`, so the
  water-near-packer interface remains conforming/shared and no
  `.pocket_packer_near` region is created; removed both pocket boundary
  conditions; prescribed `water_source_density: 1e-3` on the whole `water`
  region. All other material, hydraulic, contact, Biot and solver settings
  are unchanged (common E=60 GPa, alpha=1, conductivity=1, storativity=1).
- Mesh generation completed with 1,668 nodes, 11,459 elements and Gmsh's
  `No ill-shaped tets` report. The resulting physical names contain
  `.pocket_water` as an un-split geometric boundary label but no
  `.pocket_packer_near` and no copied interface nodes.
- Log: [ultimate_nopocket](hm_stability/ultimate_nopocket/workdir/output/flow123.0.log).
- Status: converged at t=1 in three HM iterations; no reported failures.
- HM differences (abs, rel): (14.4701, 0.104607), (138.329, 0.916479),
  (1.174e-5, 8.11328e-7).
- Comparison with S27 (pocket split): (14.5633, 0.10528), (138.329, 0.91603),
  (1.17421e-5, 8.0628e-7). The nearly identical result excludes the pocket
  split/duplicated-interface mesh as the cause of the observed HM behavior.

## S18–S29 — homogeneous hydraulic controls

All cases use the constant-fracture mesh, contact, `init_dt=end_time=1`,
explicit `.pocket_water` Dirichlet head -17.5011708 m (initial head +10 m),
conductivity=1 and storativity=1 in every hydraulic region, nu=0.25 and zero
initial stress in every mechanical region. Unless stated otherwise,
`iteration_parameter=1` and the accepted inner solver tolerances are unchanged.
Interrupted runs were stopped only after the requested trace was obtained; no
negative inner-solver convergence reason was reported.

### S18: E=1 kPa, Biot alpha=1

- Log: [ultimate_biot_1](hm_stability/ultimate_biot_1/workdir/output/flow123.0.log)
- Status: divergent at t=1; interrupted after iteration 18.
- First ten (abs, rel): (62.1684, 0.449423), (1131.63, 5.77472),
  (2130.53, 1.85777), (1220.29, 0.375883), (4287.55, 1.0302),
  (11077.5, 5.50045), (14956.6, 1.32014), (9950.29, 0.384388),
  (12987.6, 0.373623), (40511.7, 1.58349).

### S19: E=1 kPa, Biot alpha=0.2

- Log: [ultimate_biot_0p2](hm_stability/ultimate_biot_0p2/workdir/output/flow123.0.log)
- Status: slowly decreasing at t=1; interrupted after iteration 22, where
  (abs, rel)=(4.92186, 0.0257797), far from the HM tolerance.
- First ten (abs, rel): (18.1141, 0.130949), (143.824, 0.932787),
  (31.6427, 0.558126), (8.79907, 0.102356), (6.28417, 0.0664174),
  (6.10868, 0.0605809), (6.0694, 0.0567833), (6.02925, 0.0534008),
  (5.98414, 0.050334), (5.93412, 0.0475373).

### S20: E=1 kPa, Biot alpha=0.02

- Log: [ultimate_biot_0p02](hm_stability/ultimate_biot_0p02/workdir/output/flow123.0.log)
- Status: stagnates at t=1; interrupted after iteration 23.
- First ten (abs, rel): (14.6001, 0.105546), (138.289, 0.915565),
  (0.282184, 0.0192619), (0.0673512, 0.00460292),
  (0.0671174, 0.00458418), (0.0671228, 0.0045817),
  (0.0671284, 0.00457914), (0.0671339, 0.00457649),
  (0.0671394, 0.00457375), (0.0671451, 0.00457093).

### S21–S25: stabilization sweep in the soft material

- S21: alpha=0.2, multiplier=10. [Log](hm_stability/ultimate_biot_0p2_stab_10/workdir/output/flow123.0.log).
  First ten: (41.2964, 0.298537), (135.166, 0.769536),
  (51.5663, 0.682671), (23.2486, 0.231277), (11.6815, 0.0968522),
  (7.8326, 0.0594623), (6.63082, 0.047561), (6.2367, 0.0427197),
  (6.07623, 0.0399251), (5.97932, 0.0377874). Interrupted at iteration
  29: (3.5197, 0.0139304).
- S22: alpha=0.2, multiplier=100. [Log](hm_stability/ultimate_biot_0p2_stab_100/workdir/output/flow123.0.log).
  First ten: (94.9704, 0.686553), (85.4858, 0.371855),
  (60.8212, 0.35704), (59.5421, 0.382602), (56.4625, 0.323753),
  (51.2082, 0.242837), (45.1123, 0.179044), (39.0506, 0.133961),
  (33.4677, 0.10227), (28.5432, 0.0795918). A longer rerun reached its
  minimum at iteration 31, (1.28163, 0.00233727), then grew monotonically to
  (8.00993, 0.0159834) at iteration 45; it does not converge.
- S23: alpha=1, multiplier=10. [Log](hm_stability/ultimate_biot_1_stab_10/workdir/output/flow123.0.log).
  First ten: (110.732, 0.800495), (391.544, 1.58534),
  (892.52, 1.94644), (1377.37, 1.04827), (1725.6, 0.643756),
  (1752.26, 0.398671), (1248.16, 0.20388), (1240.12, 0.171124),
  (4250.64, 0.621496), (9636.7, 2.18344). Divergent; interrupted at 27.
- S24: alpha=0.2, multiplier=0. [Log](hm_stability/ultimate_biot_0p2_stab_0/workdir/output/flow123.0.log).
  First ten: (14.5633, 0.10528), (144.376, 0.956076),
  (29.4533, 0.533934), (7.38294, 0.0880792), (6.12836, 0.0672797),
  (6.09097, 0.0626973), (6.06055, 0.058738), (6.02241, 0.0551574),
  (5.9794, 0.0519227), (5.93157, 0.0489827). Interrupted at 18.
- S25: alpha=0.2, multiplier=0.1. [Log](hm_stability/ultimate_biot_0p2_stab_0p1/workdir/output/flow123.0.log).
  First ten: (14.9305, 0.107935), (144.329, 0.953695),
  (29.6554, 0.536201), (7.522, 0.0895029), (6.13879, 0.0671299),
  (6.09213, 0.0624717), (6.0614, 0.0585357), (6.02309, 0.0549762),
  (5.97987, 0.051759), (5.93181, 0.048834). Interrupted at 18.

### S26: soft alpha=0.02 with mechanics r_tol=1e-12

- Log: [ultimate_biot_0p02_mech_1e12](hm_stability/ultimate_biot_0p02_mech_1e12/workdir/output/flow123.0.log)
- Status: identical to S20 to printed precision, so the plateau is not an
  inexact mechanics-solve floor. Interrupted after iteration 17.
- First ten (abs, rel): (14.6001, 0.105546), (138.289, 0.915565),
  (0.282184, 0.0192619), (0.0673512, 0.00460292),
  (0.0671174, 0.00458418), (0.0671228, 0.0045817),
  (0.0671284, 0.00457914), (0.0671339, 0.00457649),
  (0.0671394, 0.00457375), (0.0671451, 0.00457093).

### S27–S29: E=60 GPa in every material

- S27: alpha=1. [Log](hm_stability/ultimate_stiff_biot_1/workdir/output/flow123.0.log).
  Converged at t=1 in three iterations: (14.5633, 0.10528),
  (138.329, 0.91603), (1.17421e-5, 8.0628e-7).
- S28: alpha=0.2. [Log](hm_stability/ultimate_stiff_biot_0p2/workdir/output/flow123.0.log).
  Converged at t=1 in three iterations: (14.5633, 0.10528),
  (138.329, 0.91603), (4.69684e-7, 3.22512e-8).
- S29: alpha=0.02. [Log](hm_stability/ultimate_stiff_biot_0p02/workdir/output/flow123.0.log).
  Converged at t=1 in three iterations: (14.5633, 0.10528),
  (138.329, 0.91603), (4.69609e-9, 3.22461e-10).

- Interpretation: after removing every hydraulic contrast, E=1 kPa remains
  unstable or impractically slow for nonzero alpha, while E=60 GPa is accepted
  by the HM relative tolerance for alpha=1. The final update scales as alpha
  squared. S30-S32 below show that this is tolerance-masked drift, not true
  contraction to zero. The stabilization sweep does not repair the soft
  alpha=1 case, and a tighter mechanics solve does not change the small-alpha
  plateau.

### S30-S32: stiffness bracket at Biot alpha=1

- S30: common E=10 MPa. [Log](hm_stability/ultimate_e_1e7/workdir/output/flow123.0.log).
  First ten: (14.5725, 0.105346), (138.319, 0.915913),
  (0.070479, 0.0048329), (0.0167975, 0.00115238),
  (0.0167829, 0.00115126), (0.0167833, 0.00115117),
  (0.0167836, 0.00115107), (0.016784, 0.00115097),
  (0.0167843, 0.00115087), (0.0167847, 0.00115077). The update then
  grows slightly; interrupted at iteration 20.
- S31: common E=100 MPa. [Log](hm_stability/ultimate_e_1e8/workdir/output/flow123.0.log).
  First ten: (14.5642, 0.105286), (138.328, 0.916019),
  (0.00704548, 0.000483719), (0.00167836, 0.000115237),
  (0.00167822, 0.000115226), (0.00167822, 0.000115225),
  (0.00167823, 0.000115224), (0.00167823, 0.000115224),
  (0.00167824, 0.000115223), (0.00167824, 0.000115222). Plateau persists
  through iteration 29.
- S32: common E=1 GPa. [Log](hm_stability/ultimate_e_1e9/workdir/output/flow123.0.log).
  First ten: (14.5634, 0.10528), (138.329, 0.916029),
  (0.000704528, 4.83763e-5), (0.000167822, 1.15235e-5), followed by
  (0.000167820, 1.15234e-5) through iteration 10. The same plateau persists
  through iteration 34.
- The plateau magnitudes are in the exact ratio 100:10:1 for E=10 MPa,
  100 MPa and 1 GPa. Extrapolation to E=60 GPa puts the plateau below the HM
  `r_tol=1e-6`; therefore S27-S29 stop because the drift is smaller than the
  acceptance threshold, not because the fixed-point update vanished.


## S17 — uniform mechanical stiffness

- Changes: S14 plus E=60 GPa and nu=0.25 in every material, retaining the
  original hydraulic values, contact and alpha=0.2.
- Log: [uniform_stiffness](hm_stability/uniform_stiffness/workdir/output/flow123.0.log).
- Status: timeout (180 seconds, exit 124), first step t=1, growing absolute
  differences. No negative linear convergence reason reported.
- First ten (abs, rel): (110.514, 0.798919), (145.908, 0.60062), (54.5273, 0.421559), (44.5868, 0.274636), (50.3253, 0.254932), (57.2062, 0.237556), (64.8012, 0.221276), (73.2148, 0.206978), (82.5217, 0.194666), (92.7972, 0.184104).
- Last completed iteration: 96, 4.77794e+06, 0.145283 (iteration, abs, rel).



## S14–S16 — explicit pressure and coupling localization

All stopped by internal timeout (exit 124) at the first step t=1.
No negative linear convergence reasons were found. All retain contact.

### S14: S08 plus bc_type=dirichlet on .pocket_water

- Log: [pocket_dirichlet](hm_stability/pocket_dirichlet/workdir/output/flow123.0.log)
- First ten (abs, rel): (120.401, 0.870394), (234.112, 0.911099), (255.799, 0.923115), (194.258, 0.384304), (138.046, 0.20036), (118.713, 0.145727), (132.281, 0.14403), (166.107, 0.160306), (210.717, 0.177075), (262.025, 0.188403).
- Last completed iteration: 37, 1.1532e+07, 0.562187 (iteration, abs, rel).

### S15: S14 plus rock conductivity 1e-13 → 1e-6 m/s (fracture value)

- Log: [high_matrix_k](hm_stability/high_matrix_k/workdir/output/flow123.0.log)
- First ten (abs, rel): (16.914, 0.122274), (138.312, 0.99557), (2.64682, 0.148968), (0.169182, 0.00924701), (0.102991, 0.00561423), (0.100607, 0.00547377), (0.100627, 0.00546446), (0.100802, 0.0054635), (0.100989, 0.00546303), (0.101177, 0.00546247).
- Last completed iteration: 93, 0.116817, 0.00500732 (iteration, abs, rel).

### S16: S14 plus alpha=0 in fractures/water; alpha=0.2 in rock/packers

- Log: [biot_rock_only](hm_stability/biot_rock_only/workdir/output/flow123.0.log)
- First ten (abs, rel): (110.512, 0.798908), (155.554, 0.640347), (49.5263, 0.40588), (29.5926, 0.199506), (27.3331, 0.161125), (27.0843, 0.141563), (27.0755, 0.12649), (26.9994, 0.113628), (27.0024, 0.103149), (27.0824, 0.0945424).
- Last completed iteration: 59, 6793.43, 0.188719 (iteration, abs, rel).


Inner flow/mechanics solver settings are the strict/default accepted values.
Numbers below are parsed from `output/flow123.0.log`.

## S01 — baseline

- Changes from current case: none.
- Status: diverges at the first time step (`t=1`, then reaches HM `max_it`).
- The mechanics linear solve reports reason `-3` at HM iteration 1 (1001
  iterations); later mechanics solves report positive reasons, while the outer
  HM iteration still diverges.
- HM iterations 1–10 (`abs`, `rel`):
  1. `(133.968, 0.968470)`
  2. `(362.251, 1.331900)`
  3. `(1254.67, 3.368710)`
  4. `(5239.29, 3.295020)`
  5. `(22611.4, 3.320010)`
  6. `(98091.6, 3.334430)`
  7. `(424299, 3.327620)`
  8. `(1.83325e6, 3.322270)`
  9. `(7.92330e6, 3.322060)`
  10. `(3.42471e7, 3.322270)`

## S02 — constant fracture properties

- Changes from current case: every fracture `conductivity` and `cross_section`
  `FieldFormula` was replaced by its constant replacement value (`1e-6` and
  `1e-3`, respectively). No other parameter was changed.
- Status: does not converge at the first time step, but is much less unstable
  than S01. The relative difference initially falls, reaches its minimum at
  iteration 7, then increases.
- HM iterations 1–10 (`abs`, `rel`):
  1. `(120.599, 0.871828)`
  2. `(233.263, 0.906232)`
  3. `(255.677, 0.922503)`
  4. `(196.007, 0.387951)`
  5. `(142.134, 0.205834)`
  6. `(125.156, 0.152517)`
  7. `(141.219, 0.151578)`
  8. `(177.658, 0.167703)`
  9. `(224.849, 0.183470)`
  10. `(278.847, 0.193511)`

## S03 — constant fractures, Biot coupling disabled

- Changes from current case: S02 constant fracture properties and
  `biot_alpha: 0`.
- Status: converged through time step 18 (`t=3555.4`) before the bounded run
  was stopped; no calculation failure was observed.
- First time step HM iterations (`abs`, `rel`):
  1. `(114.118, 0.824973)`
  2. `(138.329, 0.553190)`
  3. `(0, 0)`
- Subsequent steps converged in two HM iterations; for example, at `t=2.5987`:
  `(22.4764, 0.196958)`, then `(3.75909e-10, 3.63394e-12)`.

## S04 — constant fractures, contact disabled

- Changes from current case: S02 constant fracture properties and
  `mechanics_equation.contact: false`.
- Status: not converged during the bounded run; relative difference fell only
  to `0.136903` by iteration 13, so disabling contact does not remove the
  instability.
- HM iterations 1–10 (`abs`, `rel`):
  1. `(120.599, 0.871828)`
  2. `(234.823, 0.912292)`
  3. `(256.296, 0.923018)`
  4. `(192.188, 0.379647)`
  5. `(136.219, 0.198151)`
  6. `(118.891, 0.146752)`
  7. `(128.506, 0.140830)`
  8. `(148.844, 0.145413)`
  9. `(170.096, 0.147316)`
  10. `(190.041, 0.145514)`

## S05 — constant fractures, reduced nonzero Biot coupling

- Changes from current case: constant fracture properties and `biot_alpha: 0.1`.
- Status: not converged. The relative difference decreases to about `0.0292`
  at iteration 25, then rises; the absolute difference bottoms out at
  `15.3694` at iteration 17 and then rises.
- HM iterations 1–10 (`abs`, `rel`): `(116.736, 0.843903)`,
  `(142.273, 0.562255)`, `(79.4132, 0.489376)`, `(50.8137, 0.238568)`,
  `(36.2069, 0.141508)`, `(29.4881, 0.102329)`, `(25.7545, 0.0817465)`,
  `(22.9593, 0.0677785)`, `(20.636, 0.0573329)`, `(18.6916, 0.0493216)`.

## S06 — constant fractures, small nonzero Biot coupling

- Changes from current case: constant fracture properties and `biot_alpha: 0.02`.
- Status: not converged during the bounded run. The relative difference is
  monotonically decreasing, but is still `0.0149797` at iteration 27.
- HM iterations 1–10 (`abs`, `rel`): `(114.251, 0.825934)`,
  `(137.130, 0.548060)`, `(3.55883, 0.0308143)`, `(2.34502, 0.0203385)`,
  `(2.29975, 0.0198697)`, `(2.28473, 0.0196563)`, `(2.27006, 0.0194411)`,
  `(2.24938, 0.0191704)`, `(2.23549, 0.0189534)`, `(2.22179, 0.0187346)`.

## S07 — constant fractures, storage contrast removed

- Changes from current case: constant fracture properties; rock and packer
  storativity set to the water/fracture value (`4.5e-6 m^-1`); `biot_alpha: 0.2`.
- Status: not converged during the bounded run, but more contractive than S02:
  the relative difference is still decreasing (`0.0314957` at iteration 24).
- HM iterations 1–10 (`abs`, `rel`): `(130.133, 0.940749)`,
  `(140.372, 0.523695)`, `(64.4001, 0.412810)`, `(47.8982, 0.252236)`,
  `(43.2462, 0.191031)`, `(42.2674, 0.160705)`, `(42.2143, 0.140424)`,
  `(42.2811, 0.124621)`, `(42.3327, 0.111771)`, `(42.3934, 0.101228)`.


## S08–S13 — source-guided stabilization and material tests

All use the identical S02 mesh and constants, contact=true, alpha=0.2,
strict original inner solvers, end_time=1. These retain S02's missing pocket
boundary type for controlled comparisons; S14 below corrects it.
All six stopped by the internal 180-second timeout (exit 124), still at t=1;
this is not a solver-reported max_it failure. No negative linear convergence
reason appears in these six logs.

### S08: multiplier 1 (control)

- Log: [stab_1](hm_stability/stab_1/workdir/output/flow123.0.log)
- First ten (abs, rel): (120.599, 0.871828), (233.263, 0.906232), (255.677, 0.922503), (196.007, 0.387951), (142.134, 0.205834), (125.156, 0.152517), (141.219, 0.151578), (177.658, 0.167703), (224.849, 0.18347), (278.847, 0.193511).
- Last completed iteration: 34, 3.35714e+06, 0.564221 (iteration, abs, rel).

### S09: iteration_parameter=2

- Log: [stab_2](hm_stability/stab_2/workdir/output/flow123.0.log)
- First ten (abs, rel): (122.851, 0.888103), (183.056, 0.704302), (224.461, 0.947724), (227.04, 0.538808), (200.563, 0.315382), (172.68, 0.208463), (158.54, 0.159854), (165.295, 0.145277), (192.488, 0.149464), (237.957, 0.162435).
- Last completed iteration: 34, 1.22224e+06, 0.494843 (iteration, abs, rel).

### S10: iteration_parameter=10

- Log: [stab_10](hm_stability/stab_10/workdir/output/flow123.0.log)
- First ten (abs, rel): (126.597, 0.915183), (118.291, 0.44802), (112.345, 0.55046), (137.551, 0.539682), (159.554, 0.437786), (178.112, 0.349555), (194.075, 0.28599), (207.555, 0.239698), (219.039, 0.205157), (233, 0.181884).
- Last completed iteration: 35, 91740.1, 0.335722 (iteration, abs, rel).

### S11: fracture E=60 GPa; water E remains 10 MPa

- Log: [stiff_fractures](hm_stability/stiff_fractures/workdir/output/flow123.0.log)
- First ten (abs, rel): (114.262, 0.826013), (146.952, 0.587286), (68.2452, 0.494042), (62.1506, 0.342585), (71.8148, 0.309036), (83.4267, 0.281185), (96.6716, 0.257878), (111.799, 0.239015), (129.041, 0.223751), (148.653, 0.211278).
- Last completed iteration: 75, 1.61589e+06, 0.199344 (iteration, abs, rel).

### S12: every domain's storativity multiplied by 100, preserving ratios

- Log: [storage_x100](hm_stability/storage_x100/workdir/output/flow123.0.log)
- First ten (abs, rel): (133.094, 0.962152), (137.6, 0.507549), (8.50569, 0.0632625), (6.15267, 0.0455821), (5.95878, 0.0437619), (5.96924, 0.0433701), (6.00616, 0.0430989), (6.04389, 0.0427658), (6.08184, 0.0423728), (6.11325, 0.0418798).
- Last completed iteration: 50, 7.11926, 0.0213604 (iteration, abs, rel).

### S13: water E=60 GPa; fracture E remains 10 MPa

- Log: [stiff_water](hm_stability/stiff_water/workdir/output/flow123.0.log)
- First ten (abs, rel): (120.598, 0.87182), (232.531, 0.903393), (254.252, 0.924974), (197.082, 0.392902), (148.99, 0.216389), (138.935, 0.168061), (156.702, 0.164232), (186.26, 0.169378), (217.884, 0.1709), (249.723, 0.168536).
- Last completed iteration: 53, 1.06547e+10, 0.563473 (iteration, abs, rel).
