"""BGEM HealMesh quality postprocessing, preserving the unhealed mesh."""
from contextlib import redirect_stdout
import json
import logging
from pathlib import Path
import time

import gmsh
import numpy as np
from mesh_verify import tetra_face_index
from mesh_output import write_msh22
from bgem.gmsh.heal_mesh import HealMesh


def validate_gamma(gamma):
    if not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0, 1], or null to disable postprocessing')


def _quality(healer, gamma):
    histogram, bins, bad = healer.quality_statistics(bad_el_tol=gamma)
    return dict(below_gamma=len(bad), histogram=histogram.tolist(),
                upper_bounds=[float(x) if x != float('inf') else None for x in bins])


def heal_mesh_file(mesh_file: Path, gamma: float | str):
    """Run HealMesh using gamma_tol=gamma and write a separate *_healed.msh2.

    Gamma is HealMesh's normalized quality measure, not Gmsh minSICN.
    Healing may change geometry/connectivity; CAD area checks apply to the raw file.
    """
    # Bare scientific notation in YAML can arrive as a string, including in
    # --heal-existing, which bypasses MeshConfig normalization.
    gamma = float(gamma)
    validate_gamma(gamma)
    if gmsh.isInitialized():
        raise RuntimeError('Close the active Gmsh session before running HealMesh')
    mesh_file = Path(mesh_file).resolve()
    output = mesh_file.with_name(mesh_file.stem+'_healed.msh2')
    report_path = output.with_suffix('.healing.json')
    report = dict(status='healing', input=str(mesh_file), output=str(output),
                  gamma=gamma, node_tol=0.1*gamma)
    report_path.write_text(json.dumps(report, indent=2)+'\n')
    started = time.perf_counter()
    try:
        with output.with_suffix('.healing.log').open('w') as log, redirect_stdout(log):
            _phase(report, report_path, 'reading', started)
            healer = HealMesh.read_mesh(str(mesh_file), node_tol=report['node_tol'])
            report['nodes_before'] = len(healer.mesh.nodes)
            report['elements_before'] = len(healer.mesh.elements)
            report['tetrahedra_before'] = sum(e[0] == 4 for e in healer.mesh.elements.values())
            _phase(report, report_path, 'quality_before', started)
            report['quality_before'] = _quality(healer, gamma)
            _phase(report, report_path, 'repair', started)
            healer.heal_mesh(gamma_tol=gamma)
            _phase(report, report_path, 'quality_after', started)
            report['quality_after'] = _quality(healer, gamma)
            report['nodes_after'] = len(healer.mesh.nodes)
            report['elements_after'] = len(healer.mesh.elements)
            report['physical_names'] = list(healer.mesh.physical)
            _phase(report, report_path, 'writing', started)
            write_msh22(healer.mesh, output)
            del healer
        with output.open() as stream:
            if (stream.readline().strip(), stream.readline().strip()) != ('$MeshFormat', '2.2 0 8'):
                raise ValueError('HealMesh output is not ASCII GMSH 2.2')
        _phase(report, report_path, 'auditing', started)
        report['structural_check'] = _audit_output(output)
        if report['tetrahedra_before'] and not report['structural_check']['tetrahedra']:
            report['structural_check']['passed'] = False
        report['physical_names'] = report['structural_check']['physical_names']
        if not report['structural_check']['passed']:
            raise ValueError('HealMesh output has invalid tetrahedra/coordinates or detached/nonconforming surface triangles; raw mesh preserved')
        report.update(status='completed', stage='completed')
        return report
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        try:
            report['elapsed_seconds'] = time.perf_counter()-started
            report_path.write_text(json.dumps(report, indent=2)+'\n')
        finally:
            if gmsh.isInitialized():
                gmsh.finalize()


def _audit_output(output):
    """Check serialized healed topology without imposing unchanged CAD geometry."""
    gmsh.initialize()
    gmsh.option.setNumber('General.Terminal', 0)
    try:
        gmsh.open(str(output))
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        tags, xyz = tags[order], np.asarray(xyz).reshape(-1, 3)[order]
        tetra = gmsh.model.mesh.getElementsByType(4)[1].reshape(-1, 4)
        invalid_coordinates = int(np.count_nonzero(~np.all(np.isfinite(xyz), axis=1)))
        nonpositive, nonfinite, volume = 0, 0, 0.0
        for start in range(0, len(tetra), 200_000):
            points = xyz[np.searchsorted(tags, tetra[start:start+200_000])]
            with np.errstate(over='ignore', invalid='ignore'):
                volumes = np.linalg.det(points[:, 1:]-points[:, :1])/6
            nonpositive += int(np.count_nonzero(volumes <= 0))
            nonfinite += int(np.count_nonzero(~np.isfinite(volumes)))
            volume += float(volumes.sum())
        detached, bad_boundary = 0, 0
        if len(tetra):
            with tetra_face_index(tetra, directory=output.parent) as index:
                for dim, physical in gmsh.model.getPhysicalGroups(2):
                    name = gmsh.model.getPhysicalName(dim, physical)
                    for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, physical):
                        tri = gmsh.model.mesh.getElementsByType(2, int(entity))[1].reshape(-1, 3)
                        incidence = index.incidence(tri)
                        detached += int(np.count_nonzero(incidence == 0))
                        if name.startswith('.side_'):
                            bad_boundary += int(np.count_nonzero(incidence != 1))
        return dict(passed=not (invalid_coordinates or nonfinite or nonpositive or detached or bad_boundary) and bool(np.isfinite(volume)),
                    tetrahedra=len(tetra), bulk_volume=volume if np.isfinite(volume) else None,
                    nonfinite_coordinate_nodes=invalid_coordinates, nonfinite_tetrahedra=nonfinite,
                    nonpositive_tetrahedra=nonpositive, detached_surface_triangles=detached,
                    invalid_boundary_triangles=bad_boundary,
                    physical_names=[gmsh.model.getPhysicalName(d, p) for d, p in gmsh.model.getPhysicalGroups()])
    finally:
        gmsh.finalize()


def _phase(report, path, stage, started):
    report.update(stage=stage, elapsed_seconds=time.perf_counter()-started)
    path.write_text(json.dumps(report, indent=2)+'\n')
    logging.getLogger(__name__).info('HealMesh: %s', stage)
