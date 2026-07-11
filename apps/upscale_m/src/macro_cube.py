"""
MacroCube: axis-aligned cubic averaging window for the endorse homogenisation machinery.

Analogous to MacroSphere (src/endorse/homogenisation.py) with two changes:

1. The window is a max-norm ball (a cube) instead of a Euclidean ball:
       sphere:  |nc|_2   < r
       cube:    |nc|_inf < h
2. interact returns a VOLUME-FRACTION weight in [0, 1] instead of the 0/1 any-node indicator:
   the fraction of the micro element's volume lying inside the window. It is estimated with
   endorse's own uniform element refinement (refine_barycenters, src/endorse/macro_flow_model.py):
   the element is split into 8^level equal-volume sub-tetrahedra and the weight is the fraction
   of their barycenters inside the cube. The error is O(2^-level) and only for elements actually
   cut by the window boundary.

Why the fraction: on a mesh that does NOT conform to the window boundary (the general macro-mesh
setting) elements straddle the boundary, and an all-or-nothing weight misclassifies up to one
element-layer of volume. Downstream, Subdomain.weights multiplies intersect_weights by element
volumes and normalizes, so the fraction is exactly the right input: weight * V_e = overlap volume.
On a CONFORMING mesh every element is entirely in or out, so interact returns exactly 1.0 or 0.0
and the exact single-cube behaviour is recovered for free.

NOTE: MacroSphere stores `rel_radius` but its _center_radius() ignores it (r = mean vertex
distance, the attribute is dead). MacroCube applies its `rel_size` as documented:
h = rel_size * mean vertex distance. The discrepancy is recorded in PLAN.md questions.
"""
import attrs
import numpy as np

from endorse.homogenisation import MacroShapeBase
from endorse.macro_flow_model import refine_barycenters


@attrs.define
class MacroCube(MacroShapeBase):
    # cube half-edge relative to the mean distance of macro-element vertices from the barycenter
    rel_size: float
    # refinement level of the volume-fraction estimate: 8^level equal-volume sub-elements
    # (boundary error ~ element_size / 2^level); level 0 = plain barycenter 0/1 indicator
    level: int = 2

    def _center_half_edge(self, macro_el):
        center = macro_el.barycenter()
        distances = np.linalg.norm(macro_el.vertices() - center[None, :], axis=1)
        h = self.rel_size * float(np.mean(distances))
        return center, h

    def aabb(self, macro_el):
        center, h = self._center_half_edge(macro_el)
        return np.array([center - h, center + h])

    def interact(self, macro_el, micro_el):
        """Volume fraction of the micro element inside the cubic window, in [0, 1]."""
        center, h = self._center_half_edge(macro_el)
        nodes = micro_el.vertices() - center[None, :]

        # exact cheap cases: all nodes inside -> whole element inside (window is convex);
        # node AABB disjoint from the window on some axis -> no overlap at all
        node_inf = np.max(np.abs(nodes), axis=1)
        if np.all(node_inf < h):
            return 1.0
        if np.any(np.min(nodes, axis=0) > h) or np.any(np.max(nodes, axis=0) < -h):
            return 0.0

        # element is (possibly) cut by the window boundary: equal-volume subdivision estimate
        barycenters = refine_barycenters(nodes, self.level)
        inside = np.max(np.abs(barycenters), axis=1) < h
        return float(np.mean(inside))
