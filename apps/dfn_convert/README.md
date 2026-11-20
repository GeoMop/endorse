# DFN conversion

This application converts a discrete fracture network (DFN) exported by
DFN_gen into a mesh for ENDORSE/Flow123d simulations. A DFN describes individual
fractures inside a rock volume. The converter reconstructs their polygons,
clips them to a box, and meshes the fractures together with the surrounding
rock. Fracture triangles and rock tetrahedra share nodes, so the resulting mesh
represents their intersections explicitly.

The application prepares geometry and meshes; it does not generate a new random
fracture network or run a flow or mechanical simulation. DFN datasets are
external inputs, supplied separately from the code.

## Installation

Run the following commands from `apps/dfn_convert`, using Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The requirements pin the BGEM `JB_homo` revision and Gmsh version used by this
application. The older BGEM in `../../submodules/bgem` has a different API and
cannot replace that dependency. For tests and optional VTP comparison, install
the development requirements as well:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt
```

## Input

Supply a DFN_gen export directory containing:

- `DFN_gen-vars.yaml`: domain bounds, fracture-family names, and their associations;
- `ellipses_sSS_dDDD_rRRR.yaml`: metadata and array names for each selected family;
- matching `ellipses_sSS_dDDD_rRRR.npz` files: fracture parameters stored as named arrays;
- optionally, `fractures_rRRR.vtp` for comparison with the reconstructed polygons.

Here `SS`, `DDD`, and `RRR` are the zero-padded family, domain, and realization
indices. The loader selects one domain and realization, and reads the families
listed for that domain. It supports cuboid domains and reconstructs the source
polygon sampling, including non-circular polygons, without resampling fractures.
Source filenames and row numbers are retained for tracing mesh surfaces back to
the input.

## Configuration and first run

Create a configuration such as `my-config.yaml`, pointing to your external
input directory:

```yaml
input:
  directory: /path/to/dfn-export
  domain_idx: 0
  realization: 0
output: output/network.msh2
coordinate_scale: 1.0
mesh:
  fracture_size: 0.05
  bulk_size: 0.25
  tolerance: 1.0e-8
  threads: 1
  occ_parallel: false
  refinement_distance: 0.1
  gamma: null
```

Start by inspecting the input, then generate a mesh:

```bash
.venv/bin/python convert.py --config my-config.yaml --summary-only
.venv/bin/python convert.py --config my-config.yaml
```

The first command prints fracture counts, areas, bounds, and size statistics,
and writes a summary JSON file. It does not run CAD construction or meshing and
does not require BGEM or Gmsh. Add `--check-vtp` to compare the complete input
network with its VTP export before any cropping; this requires PyVista.

Without `--config`, the command reads the adjacent `config.yaml`. Set its paths
for your own input, or override them on the command line:

```bash
.venv/bin/python convert.py --input /path/to/dfn-export --output output/network.msh2
```

The `input` mapping groups `directory`, `domain_idx`, `realization`, `crop`,
`rmin`, and `rmax`. `output`, `coordinate_scale`, and `mesh` remain top-level
settings. Paths in a configuration resolve relative to that configuration file.
Command-line `--input` and `--output` paths resolve relative to the current
working directory.

For an initial mesh, select a small box **within your source domain**:

```bash
.venv/bin/python convert.py --config my-config.yaml \
  --crop -0.25 0.25 -0.25 0.25 -0.25 0.25 --output output/crop.msh2
```

The six values are `X0 X1 Y0 Y1 Z0 Z1`. Cropping includes polygons that cross the
box even when their centers lie outside it; exact clipping removes portions
outside the box. An `input.crop` list with the same six values can also be set in YAML.

`coordinate_scale` multiplies source length quantities and domain bounds
consistently. The converter does not infer physical units. Crop coordinates,
radius limits, mesh sizes, and tolerances must use the resulting coordinate system.

Select fractures by radius using optional settings under `input`:

```yaml
input:
  directory: /path/to/dfn-export
  rmin: 0.05
  rmax: 0.2
```

Equivalent CLI options are `--rmin 0.05 --rmax 0.2`; each overrides its YAML
setting. Either bound can be omitted (or set to `null` in YAML), and both bounds
are inclusive. Radius is the input `fracture_req`, the area-equivalent radius
`sqrt(source_polygon_area / pi)`, in scaled coordinate units. It is evaluated
before cropping, rather than recomputed from the clipped polygon. Filtering
preserves family names and original source-row IDs, and applies to meshing,
input summaries, and BGEM exports. `--check-vtp` still checks the complete source
before filtering. An interval retaining no fractures generates only the bulk
mesh. The optional `radius_filter` summary entry records the requested bounds.

## Choosing mesh settings

`fracture_size` is the target element size on fractures; `bulk_size` is the
target away from them. With a positive `refinement_distance`, the mesh transitions
from the fracture size to the bulk size over that distance. Setting it to zero
uses basic point-size interpolation, which can spread fine elements throughout
a densely fractured volume. Small fracture intersections can force elements
smaller than either target.

`mesh.geometry_tol` enables OCC CAD healing after fragmentation and before
meshing when positive; its default is `0.0` (disabled). The value is in scaled
coordinate units, for example:

```yaml
mesh:
  geometry_tol: 1.0e-4
```

`--geometry-tol` overrides that value. Small-edge/face and degeneracy fixes are
enabled; sewing faces and rebuilding solids are disabled to retain fracture
sheets and bulk partitions. OCC renumbers entities, so source ownership is
reconstructed geometrically and physical fracture-family names are reassigned.
Ambiguous ownership, excessive source-area changes, and missing bulk solids
raise an error. The allowed source-area change is bounded by
`geometry_tol * source_perimeter + pi * geometry_tol**2`, plus the existing
relative numerical allowance. Each retained source records
`pre_heal_clipped_area`, its new `clipped_area`, and `geometry_area_change`.
Removed sources within that area budget are marked `healed_removed`.

The `.report.json` and `.manifest.json` contain `geometry.before_healing` and
`geometry.after_healing`. Each includes min/max/count for `area_perimeter`
(all CAD faces) and `volume_surface_area` (all CAD volumes), plus entity counts.
These are length-valued feature-size indicators. Boundary loops, including holes,
contribute to perimeters; shared faces contribute to each adjacent volume's
surface area. Metrics are reported even with healing disabled, and saved before
meshing starts. Empty collections have `null` min/max. Mesh verification still
checks the original bulk box and the adjusted CAD fracture areas.

`tolerance` controls CAD tolerances. `threads` controls Gmsh threading, and
`occ_parallel` enables OpenCASCADE parallel Boolean processing. The example
values are starting points and need to suit the input's scale and complexity.
Use `--fracture-size`, `--bulk-size`, and `--verbose` for command-line size
overrides and detailed meshing progress.

The converter clips convex polygons and fragments all fracture faces jointly
with the rock box. Positive-area coplanar overlaps between source fractures are
rejected because assigning their physical ownership requires a separate policy.
Large networks can require substantial CAD processing time, memory, and disk
space; check a crop before attempting a full conversion.

## Mesh output and verification

The main output is a first-order ASCII Gmsh 2.2 file (`.msh2`) containing:

| Physical region | Elements and meaning |
| --- | --- |
| `bulk` | Tetrahedra representing the rock volume |
| Input fracture-family names | Triangles representing fractures in each family |
| `.side_x0` through `.side_z1` | Exterior surface triangles without fracture ownership; `0` is the lower bound, `1` the upper bound |
| `.fr_outer_side_*` | Fracture traces on the exterior of the box |

The file already includes fracture and bulk elements; separate surface meshing
is unnecessary. Each fracture keeps its source identity and CAD surface tags in
the manifest, while physical regions group triangles by family.

A fracture on an exterior side retains its fracture-family label. In that case,
the corresponding `.side_*` region covers only the rest of the side and may be
absent if a fracture covers the whole side. For precise selection of exterior
fracture surfaces, use manifest records with `boundary_sides` and their
`surface_tags`. The report's `boundary_membership` lists touching physical
regions, but a listed family may also contain internal fractures.

For an output `network.msh2`, companion files are:

| File | Purpose |
| --- | --- |
| `network.summary.json` | Input statistics, crop bounds, and optional VTP comparison |
| `network.manifest.json` | Geometry-stage mapping from source fractures to clipped CAD surfaces and areas |
| `network.report.json` | Final physical IDs, software versions, timings, peak process memory, verification results, or failure stage |
| `network.gmsh.log` | Gmsh diagnostic messages |
| `network.brep` | Fragmented CAD checkpoint before element generation |
| `network.verification.json` | Verification status and results, refreshed by standalone rechecks |
| `network.failed.brep` | Diagnostic CAD geometry when geometry construction or meshing fails |

Before reporting success, the converter reopens the written mesh and checks
positive tetrahedron volumes, total rock volume, fracture areas and labels,
exterior coverage, and shared triangle/tetrahedron node IDs. Internal fracture
triangles must adjoin two tetrahedra; exterior triangles must adjoin one.
Verification uses a temporary disk-backed face table of about 96 bytes per
tetrahedron under the output directory, removed afterwards. For large meshes,
choose an output directory with sufficient disk space rather than RAM-backed
storage.

Review the Gmsh log and reported minimum `minSICN` quality for the intended
simulation. Passing geometry and connectivity checks does not guarantee the
mesh quality needed by a solver.

Recheck an existing raw mesh using its saved manifest, without rebuilding it:

```bash
.venv/bin/python convert.py --config my-config.yaml \
  --output output/network.msh2 --verify-existing
```

## Optional mesh healing

Healing attempts to repair low-quality elements after the raw mesh passes
verification. Enable it with `mesh.gamma` in YAML or `--gamma 0.0001` on the
conversion command. The default `null` disables it. Gamma must be in `(0, 1]`
and is HealMesh's normalized quality threshold, distinct from `minSICN`.
Larger values target more elements for repair; the threshold is not a guaranteed
final minimum quality.

To heal an existing mesh:

```bash
.venv/bin/python convert.py --config my-config.yaml \
  --output output/network.msh2 --heal-existing --gamma 0.0001
```

This keeps the raw file and writes `network_healed.msh2`,
`network_healed.healing.json`, and `network_healed.healing.log`. The JSON records
node and element counts and quality histograms before and after repair.

Healing can move nodes and change connectivity, so the raw manifest's CAD-area
and per-source surface checks do not apply to the healed file. Family physical
labels are retained. The healed file is reopened to check positive tetrahedra
and attachment of surface triangles to tetrahedron faces. A failed check fails
the command and retains the healed file for diagnosis; reducing gamma may help,
but no threshold guarantees a valid repair for every input.

## Python integration

The main entry points are:

- `dfn_data.load_network(Path(...))`: load numeric input independently of BGEM and Gmsh;
- `dfn_data.polygon_vertices(family)`: reconstruct source polygon vertices;
- `mesh.generate_mesh(network, MeshConfig(...), Path(...))`: generate and verify a mesh, managing its own Gmsh session;
- `mesh.build_geometry(network, config, factory)`: build geometry within a caller-managed BGEM factory;
- `bgem_adapter.to_fracture_sets(network)`: create BGEM FractureSets with source-row mappings.

The BGEM adapter represents regular source polygons with the correct scale and
first-vertex orientation. It rejects non-circular source polygons that cannot
be represented exactly by an affine regular BGEM polygon; the explicit-polygon
mesher supports those inputs.

To export BGEM arrays and source mappings without pickled Python objects:

```bash
.venv/bin/python convert.py --config my-config.yaml --summary-only \
  --bgem-output output/fracture_sets.npz
```

Each `gN_` group contains `base_shape_idx`, `radius`, `center`, `normal`,
`shape_axis`, `family`, `source`, `source_rows`, and `family_name`. Use the numeric
fields to reconstruct a `FractureSet`, keeping provenance separately.

## Development checks

```bash
.venv/bin/python -m pytest tests
.venv/bin/python -m tox
```

Tox runs the test suite with Python 3.12 and the development requirements.
Historical validation runs and implementation notes are in
[docs/validation.md](docs/validation.md) and [docs/progress.md](docs/progress.md).


HealMesh progress is recorded in the JSON `stage` field (`reading`,
`quality_before`, `repair`, `quality_after`, `writing`, `auditing`, `completed`)
and announced on stderr. Export streams ASCII GMSH 2.2 directly, preserving IDs
and tags, without rebuilding another Gmsh model element by element. A temporary
`.writing` file is replaced into the output path after export closes. This avoids
the very slow BGEM writer on large meshes; repair and quality scans still take
time proportional to the input and repairs.
