"""Exercise CAD healing and geometric length ratios with the real OCC kernel."""
import gmsh
import numpy as np
import pytest


def test_geometry_metrics_use_actual_boundary_measures():
    from geometry_healing import geometry_metrics
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    try:
        gmsh.model.occ.addBox(0, 0, 0, 2, 4, 6)
        gmsh.model.occ.synchronize()
        result = geometry_metrics()
        assert result['area_perimeter'] == dict(count=6, min=pytest.approx(2/3), max=pytest.approx(6/5))
        assert result['volume_surface_area'] == dict(count=1, min=pytest.approx(6/11), max=pytest.approx(6/11))
        assert result['entities'] == {'0': 8, '1': 12, '2': 6, '3': 1}
    finally:
        gmsh.finalize()


def test_healing_preserves_embedded_surface_ownership():
    from geometry_healing import heal_cad, geometry_metrics
    # A square with an extra vertex a micrometre along a straight edge.
    polygon = np.array([[-.3, -.3, 0], [.3, -.3, 0], [.3, -.3+1e-6, 0],
                        [.3, .3, 0], [-.3, .3, 0]])
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    try:
        gmsh.model.occ.addBox(-.5, -.5, -.5, 1, 1, 1)
        points = [gmsh.model.occ.addPoint(*p) for p in polygon]
        lines = [gmsh.model.occ.addLine(a, b) for a, b in zip(points, points[1:]+points[:1])]
        face = gmsh.model.occ.addPlaneSurface([gmsh.model.occ.addCurveLoop(lines)])
        gmsh.model.occ.fragment([(3, 1)], [(2, face)])
        gmsh.model.occ.synchronize()
        before = geometry_metrics()
        ownership = heal_cad({'source': polygon}, 1e-4)
        after = geometry_metrics()
        assert set(ownership.values()) == {'source'}
        assert len(ownership) == 1
        assert sum(gmsh.model.occ.getMass(*dt) for dt in ownership) == pytest.approx(.36)
        assert after['entities']['2'] == before['entities']['2']
        assert after['entities']['3'] == 1
        volume = gmsh.model.getEntities(3)[0]
        assert set(gmsh.model.mesh.getEmbedded(*volume)) == set(ownership)
    finally:
        gmsh.finalize()


def test_healing_distinguishes_parallel_sources_closer_than_tolerance():
    from geometry_healing import heal_cad
    polygons = {name: np.array([[-.2, -.2, z], [.2, -.2, z], [.2, .2, z], [-.2, .2, z]])
                for name, z in [('lower', 0.), ('upper', 5e-5)]}
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    try:
        bulk = gmsh.model.occ.addBox(-.5, -.5, -.5, 1, 1, 1)
        sheets = [(2, gmsh.model.occ.addRectangle(-.2, -.2, p[0, 2], .4, .4))
                  for p in polygons.values()]
        gmsh.model.occ.fragment([(3, bulk)], sheets)
        gmsh.model.occ.synchronize()
        ownership = heal_cad(polygons, 1e-4)
        assert set(ownership.values()) == {'lower', 'upper'}
        assert len(ownership) == 2
        for dt, source in ownership.items():
            assert gmsh.model.occ.getCenterOfMass(*dt)[2] == pytest.approx(polygons[source][0, 2], abs=1e-12)
    finally:
        gmsh.finalize()


def test_healing_does_not_assign_nearby_box_boundary_to_fracture():
    from geometry_healing import heal_cad
    polygon = np.array([[-.5, -.5, .49995], [.5, -.5, .49995],
                        [.5, .5, .49995], [-.5, .5, .49995]])
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    try:
        bulk = gmsh.model.occ.addBox(-.5, -.5, -.5, 1, 1, 1)
        sheet = gmsh.model.occ.addRectangle(-.5, -.5, .49995, 1, 1)
        gmsh.model.occ.fragment([(3, bulk)], [(2, sheet)])
        gmsh.model.occ.synchronize()
        ownership = heal_cad({'near_boundary': polygon}, 1e-4)
        assert len(ownership) == 1
        for dt in ownership:
            assert gmsh.model.occ.getCenterOfMass(*dt)[2] == pytest.approx(.49995, abs=1e-12)
    finally:
        gmsh.finalize()
