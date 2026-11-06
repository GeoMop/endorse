# Minimal homogenization test

## Goal

Provide a fast, deterministic problem for debugging and iterating the Stage A homogenization code.
The main case is a box with two conductivity regions and one fracture. Two simpler cases help isolate failures.
Use the existing mesh preparation and three-load calculation, with minimal test-specific code.

This document describes future implementation in `test_small.py`, `test_small.yaml`, and `test_small_tmpl.yaml`.
The current change is documentation only. See [Stage A](../PLAN.md#stage-a---deterministic-homogenization-problem).

## Geometry and calculation

- Box dimensions: 100 × 100 × 100, centred at the origin, with coordinates from -50 to 50.
- Homogenization mesh: the whole box, nominal step 20, fixed across variants.
- Micro mesh: the same box, nominal step 20, split at `x = 0` into `box_A` and `box_B`.
  Each half has dimensions 50 × 100 × 100 and centre at `x = -25` or `x = 25`.
- Start with fracture mesh step 20 as well. Mesh steps are targets, not prescribed element counts.
- Use explicit positive isotropic matrix conductivities, fracture conductivity, and fracture aperture.
  Begin with moderate contrasts; select numerical values during implementation.
- Keep fixed meshing seeds, the three coordinate pressure loads, coverage preflight, and averaging scale 2.

The fracture is a plane spanning the box, with its orientation parameterized in the **xy plane**.
Rotate about the z-axis through the fixed fracture centre `c = (0, 0, 0)`.
Define the angle in degrees and the plane by:

```text
n(theta) = (cos(theta), sin(theta), 0)
n(theta) · (r - c) = 0
```

Use `theta = 45°` for the deterministic fractured case, giving `x + y = 0`.
This is a vertical fracture whose normal rotates in the xy plane; it is not a horizontal fracture.
The offset avoids coincidence with the vertical box edges at the default angle.
Construct a sufficiently large rectangle in this plane and clip it to the box.
Keep its centre fixed when changing the angle so rotation does not translate it.

## Three shared variants

| Case | Matrix conductivity | Fracture | Purpose |
| --- | --- | --- | --- |
| 1. Uniform | Equal in both halves | None | Recover an isotropic tensor |
| 2. Two materials | Different in the two halves | None | Check material assignment and interface averaging |
| 3. Fractured | Same contrast as case 2 | One, at 45° | Exercise the complete homogenization calculation |

Case 3 is the main debugging target. Reuse one runner, geometry builder, material assignment, and validation.
Keep both material regions even in case 1; distinguish cases through material contrast and fracture inclusion.
Expose the fracture angle as a parameter from the first implementation.
Avoid a new test framework or duplicated runners.
Use distinct micro and homogenization mesh names and separate case outputs so results cannot overwrite each other.

### Future case 4: random fracture angle

Later, sample the fracture angle to exercise the sampling pipeline with this small physical problem.
Reuse the same fracture builder and homogenization runner, with the same fixed homogenization mesh.
Preserve reproducibility through explicit sampling seeds and retain the sampled angle with each result.
The angle distribution, sampling orchestration, and statistical checks are future work.
Do not implement random sampling as part of the initial three cases.

## Missing pieces in the current files

- Load `test_small.yaml` from the test directory; the current path points to `test_small.py` in the application root.
- Resolve the Flow123d template relative to a known input directory before changing the working directory.
- Construct the deterministic fracture set explicitly; it is currently undefined in the runner.
- Assign distinct mesh output names; both calls currently use the same mesh configuration and name.
- Correct the half-box rectangle centres and verify their extrusion covers exactly the intended box.
- Define matrix and fracture material properties and pass them through the existing field preparation path.
- Check material and fracture region names against the solver template, including fracture boundary regions.

## Acceptance checks and diagnostics

- Verify the two half-boxes cover the intended domain without gaps or overlapping volume.
- Verify expected material and fracture regions, distinct mesh paths, and stable homogenization element identities.
- Require nonempty averaging coverage before starting Flow123d.
- Complete all three loads and return finite six-component conductivity tensors with positive eigenvalues.
- For case 1, compare against the prescribed isotropic conductivity tensor with a documented numerical tolerance.
- For case 2, inspect material assignment and behavior near the interface. Do not require a global layered-medium
  formula to hold exactly for every local averaging domain.
- For case 3, check fracture geometry and assigned properties, and compare tensors against case 2.
- Preserve meshes, solver inputs and outputs, coverage diagnostics, tensor summaries, and stage timings.
  Report failing element identities and relevant material parameters for post-mortem inspection.

Measure geometry, meshing, preparation, and solve times before changing resolution or imposing a runtime threshold.
If needed, try coarser meshes while retaining coverage and the geometry checks.
Numerical tolerances and material values remain implementation-stage choices.
Transport, random DFN generation, and new Dask orchestration are outside the initial deterministic test.
