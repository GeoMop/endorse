# Postprocessed laser scan of L5 gallery at Bukov URF II

Processing of the laser scan point data to a tight single layer surface mesh of the 
excavations.


## Coordinate Systems
- global system is JSTK, that is system of the input data
- local system, is just shifted in X and Y, with origin at (-622600, -1127800)
- simulation system, parallel with L5

## Important Points

Point at floor at intersection of L5 axis and ZK5-1J axis:  P1 = [-66.52, -20.96, 18]
Point at floor at intersection of L5 axis and ZK5-1S axis:  P2 = [-56.80, -24.45, 18]
Origin of simulation system: O = (P1 + P2)/2 = [-61.66, -22.71]
X axis direction: [9.72, -3.49, 0] 

Floor at L5 is slightly lower at: z = 17.5

## Simulation Domain

Domain X axis lays at axis of L5, Y axis is parpendicular in the middle of the ZK5-1J and ZK5-1S.
Z==0 is at floor of ZKs at 18m of local coordinate system
Domain extension is about 12m to each direction from drifts ZK5-1J and ZK5-1S. 
box diams: [45, 45, 30]; X range from L5 stationing 15m - 60m 

## Original Data

`L5_original_data.obj` - provided by GeoTechnika, 14M vertices (JSTK), 24M triangular faces; resolution about 1cm

## Processed Data
Triangular meshes provided in VTX and GMSH 2.2 mesh formats. Conversion to other formats could be done e.g. by
[meshio](https://pypi.org/project/meshio/#description)

`L5_local.(vtx|msh)` - original mesh in local system with origin (0, 0) = (-622600, -1127800) JSTK,
                 essential to avoid round off errors in processing by tools using only single precission floats
                 (problem in MeshLab and GMSH)

`L5_local_<L>.(vtx|msh)` - reconstructed meshes with decreasing resolution about 2**L cm, resolves holes and overlapping faces 
                 probably due to stitching the scans done from different positions



