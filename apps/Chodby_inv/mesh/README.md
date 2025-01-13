# Geometries and meshes for project Chodby
(update 2025-01-10)

## Bukov II - pore pressure experiment models

### Coordinate Systems
- global system is JSTK, that is system of the input data
- local system, is just shifted in X and Y, with origin at (-622600, -1127800)
- simulation system, parallel with L5 (see bellow)

### Simulation Domain
Domain Y axis lays at axis of L5, X axis is in direction of drift ZK5-1S.
Y axis direction: [9.72, -3.49, 0] (in the local system).

Point at floor at intersection of L5 axis and ZK5-1J axis:  P1 = [-66.52, -20.96, 18] (in the local system)
Point at floor at intersection of L5 axis and ZK5-1S axis:  P2 = [-56.80, -24.45, 18] (in the local system)
Origin of simulation system: O = (P1 + P2)/2 = [-61.66, -22.71] (in the local system)
Origin coordinates are given by requirements:  X in the middle of L5, Y in the middle between ZK5-1J and ZK5-1S, Z at the floor of L5

Domain extension is about 20m to each direction, more specificaly
domain dimensions are: [50, 65, 45] with origin in the middle.
Note: floor at L5 is slightly lower at: z = 17.5 (local system)

### Original Scan Data

`scan_meshes/L5_original_data.obj` - provided by GeoTechnika, 14M vertices (JSTK), 24M triangular faces; resolution about 1cm
stored separately on the Google Drive through DVC tool.


### Processed Data
Triangular meshes provided in VTX and GMSH 2.2 mesh formats. Conversion to other formats could be done e.g. by
[meshio](https://pypi.org/project/meshio/#description)

`L5_local.(vtx|msh)` - original mesh in local system with origin (0, 0) = (-622600, -1127800) JSTK,
                 essential to avoid round off errors in processing by tools using only single precission floats
                 (problem in MeshLab and GMSH)

`L5_local_<L>.(vtx|msh)` - reconstructed meshes with decreasing resolution about 2**L cm, resolves holes and overlapping faces 
                 probably due to stitching the scans done from different positions





## Near-field transport model
