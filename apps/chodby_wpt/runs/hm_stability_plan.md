# HM stability search plan

Purpose: locate the first physical/model feature that makes the WPT HM fixed
point iteration diverge. All scenarios keep the accepted material properties,
the strict inner-solver tolerances, and the default mechanics solver.

The reference cases are Flow123d `tests/50_mechanics/01_inject.yaml` (constant
fracture properties) and `06_inject_var_cond.yaml` (mechanics-dependent
fracture conductivity). Both are passing integration tests. The WPT model is
not substituted for either test; they establish the intended progression from
constant fracture properties to added feedback.

Each scenario uses a distinct `runs/hm_stability/<name>/workdir`, starts from a
new mesh, and is recorded in `hm_stability_experiments.md`. A failed run is
recorded by the first ten `HM Iteration` absolute and relative differences from
`output/flow123.0.log`; a successful one records its last HM iteration and any
later failure time.

1. `baseline`: Current WPT template, including spatial fracture
   `FieldFormula` conductivity and cross-section.
2. `constant_fractures`: Replace only the fracture conductivity and
   cross-section `FieldFormula`s by their corresponding constants. This tests
   spatial material contrast. These formulas currently depend only on position,
   not `cross_section_updated`, so they are not themselves an HM feedback loop.
3. `biot_off`: Starting from the last stable constant-fracture case, set
   `biot_alpha: 0`. This isolates pressure-to-mechanics and volumetric-strain-
   to-flow coupling from the prescribed-pressure flow problem.
4. `no_contact`: Restore `biot_alpha`, disable mechanics contact, and keep
   constant fracture properties. This isolates contact/aperture mechanics.
5. `biot_ramp`: Restore contact and sweep `biot_alpha` from the stable zero
   case upward (0.01, 0.02, 0.05, 0.1, 0.2), with constant fracture
   properties. This brackets the coupling strength at which the fixed point
   becomes unstable.
6. `pressure_ramp`: At the largest stable Biot coefficient, apply the pocket
   head in increments (1, 10, 30, 100, 300 m) to find a load threshold.
7. Reintroduce one feature at a time: fracture spatial conductivity profile,
   spatial cross-section profile, then the production pressure history.

Only changes that move the failure boundary will be retained for a follow-up
parameter study. Solver tolerances and solver types remain fixed throughout.
