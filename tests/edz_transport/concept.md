# Fine scale local models

Model of single container position with EDZ and possible heterogenities, no fractures.
=> from application of several boundary conditions determination of ekvivalent properties at neigbourhood of given points

# Fine scale

Fully resolved center storage borehole and its EDZ, meshing governed by a distance field from the borehole surface. 
Some fractures.
Side boreholes not resolved but replaced by equivalent properties 

# First coarsening

Replace by model with tunnels replaced by equivalent properties, elements about 5m, only major fractures.

# Second coarsening

Possible replacement of major fractures. More advanced, possible usage of analytical determination of the equivalent properties around fractures.

Neccessary tools:
- mark elements of a mesh that have berycenters (or nodes) in a given shape (auxiliary meshed as well)
- assigning regions to elements after meshing
- determine effective permeability tensor and dispersion tensor from given averages, deal with nearly degenerated input vectors (3 comp for pressure, 3 comp for conc.)
  How to determine that least sqauares matrix is close to singular? That would be just a warning. determinant would be possibly good measure, but scaled by abs value of largest vector.
  But te abs value of the vecotr as well as the ratio largest/smales vector sizes ... could be good indicators of correct or corrupted homogenization.
  

- homogenisation avarages for given points and radius, from given set of evaluations, 
  test the dot product of 
- mark elements of the coarse mesh containing the fine scale models -> assing particular regions -> calculate the anisotropic tensors on these regions, pass them to Flow123d


TODO:
- large scale flow model seems to not provide reasonable 
