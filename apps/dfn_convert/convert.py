#!/usr/bin/env python3
"""Convert a DFN_gen realization to a verified, conforming MSH 2.2 mesh."""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

from dfn_data import compare_vtp, crop_network, filter_by_radius, load_network, polygon_area, polygon_vertices


log = logging.getLogger(__name__)


def summarize(network):
    families = []
    for family in network.families:
        vertices = polygon_vertices(family)
        crossing = np.any((vertices < network.bounds[:, 0]) | (vertices > network.bounds[:, 1]), axis=(1, 2))
        families.append(dict(name=family.name, source=str(family.source), count=len(family),
                             semi_major_min=float(family.arrays['fracture_aaxis'].min()),
                             semi_major_max=float(family.arrays['fracture_aaxis'].max()),
                             source_area=float(polygon_area(vertices).sum()),
                             boundary_crossing=int(crossing.sum())))
    return dict(fractures=len(network), bounds=network.bounds.tolist(),
                coordinate_scale=network.coordinate_scale, families=families)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--input', type=Path, help='Override input directory (relative to current directory)')
    parser.add_argument('--output', type=Path, help='Override mesh path (relative to current directory)')
    parser.add_argument('--crop', type=float, nargs=6, metavar=('X0', 'X1', 'Y0', 'Y1', 'Z0', 'Z1'), help='Crop in scaled coordinate units; includes polygons whose centers lie outside')
    parser.add_argument('--rmin', type=float, help='Minimum area-equivalent fracture radius, inclusive, in scaled units')
    parser.add_argument('--rmax', type=float, help='Maximum area-equivalent fracture radius, inclusive, in scaled units')
    parser.add_argument('--summary-only', action='store_true')
    parser.add_argument('--verify-existing', action='store_true', help='Recheck output against its saved CAD manifest without remeshing')
    parser.add_argument('--heal-existing', action='store_true', help='Run HealMesh on the output file without regenerating geometry')
    parser.add_argument('--gamma', type=float, help='HealMesh gamma quality threshold (overrides mesh.gamma)')
    parser.add_argument('--check-vtp', action='store_true', help='Compare the complete source to its supplied VTP before radius filtering or cropping')
    parser.add_argument('--bgem-output', type=Path, help='Save BGEM FractureSet arrays and source-row mapping as NPZ')
    parser.add_argument('--geometry-tol', type=float, help='OCC healing tolerance in scaled units; nonpositive disables CAD healing')
    parser.add_argument('--fracture-size', type=float)
    parser.add_argument('--bulk-size', type=float)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    input_config = config['input']
    source = args.input or config_path.parent / input_config['directory']
    output = (args.output or config_path.parent / config['output']).resolve()
    if args.heal_existing:
        if args.verify_existing or args.summary_only:
            parser.error('--heal-existing cannot be combined with --verify-existing or --summary-only')
        from healing import heal_mesh_file
        gamma = args.gamma if args.gamma is not None else config.get('mesh', {}).get('gamma')
        if gamma is None:
            parser.error('--heal-existing requires --gamma or mesh.gamma')
        print(json.dumps(heal_mesh_file(output, gamma), indent=2))
        return 0
    if args.verify_existing:
        from mesh import verify_mesh_file
        print(json.dumps(verify_mesh_file(output), indent=2))
        return 0
    network = load_network(source, domain_idx=input_config.get('domain_idx', 0),
                           realization=input_config.get('realization', 0),
                           coordinate_scale=config.get('coordinate_scale', 1.0))
    log.info('Loaded %d source fractures', len(network))
    vtp = compare_vtp(network, source/f'fractures_r{network.realization:03d}.vtp') if args.check_vtp else None
    rmin = args.rmin if args.rmin is not None else input_config.get('rmin')
    rmax = args.rmax if args.rmax is not None else input_config.get('rmax')
    if rmin is not None or rmax is not None:
        network = filter_by_radius(network, rmin, rmax)
        log.info('Radius filter retains %d source fractures', len(network))
    crop = args.crop or input_config.get('crop')
    if crop is not None:
        bounds = np.asarray(crop, dtype=float).reshape(3, 2)
        if np.any(bounds[:, 0] >= bounds[:, 1]) or np.any(bounds[:, 0] < network.bounds[:, 0]) or np.any(bounds[:, 1] > network.bounds[:, 1]):
            raise ValueError('Crop must have positive extents within the source domain')
        network = crop_network(network, bounds)
        log.info('Crop retains %d source polygons before exact clipping', len(network))
    summary = summarize(network)
    if rmin is not None or rmax is not None:
        summary['radius_filter'] = dict(rmin=rmin, rmax=rmax, radius_field='fracture_req')
    if vtp is not None:
        summary['vtp_comparison'] = vtp
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    if args.bgem_output:
        from bgem_adapter import to_fracture_sets
        groups = to_fracture_sets(network)
        arrays = {}
        for i, group in enumerate(groups):
            arrays.update({f'g{i}_{name}': getattr(group.fractures, name)
                           for name in ('radius', 'center', 'normal', 'shape_axis', 'family')})
            arrays.update({f'g{i}_base_shape_idx': group.fractures.base_shape_idx,
                           f'g{i}_source_rows': group.source_rows, f'g{i}_source': str(group.source),
                           f'g{i}_family_name': group.family_name})
        args.bgem_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.bgem_output, **arrays)
    if args.summary_only:
        print(json.dumps(summary, indent=2))
        return 0
    from mesh import MeshConfig, generate_mesh
    mesh_settings = config.get('mesh', {}).copy()
    mesh_settings.update({key: value for key, value in
                          [('fracture_size', args.fracture_size), ('bulk_size', args.bulk_size), ('gamma', args.gamma), ('geometry_tol', args.geometry_tol)] if value is not None})
    mesh_settings['verbose'] = args.verbose or mesh_settings.get('verbose', False)
    log.info('Generating %s', output)
    report = generate_mesh(network, MeshConfig(**mesh_settings), output)
    log.info('Verified: %s', report['verification'])
    print(report['healing']['output'] if 'healing' in report else output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
