import json
from pathlib import Path
import subprocess
import sys

import pytest


APP = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('cli_override', [False, True])
def test_cli_summary_resolves_config_paths_and_preserves_crop_identities(tmp_path, cli_override):
    config = tmp_path/'settings.yaml'
    (tmp_path/'source').symlink_to(APP/'DFN_gen-out_v01_d004', target_is_directory=True)
    config.write_text('input:\n  directory: source\n  domain_idx: 0\n  realization: 0\n'
                      '  crop: [-.2, .2, -.2, .2, -.2, .2]\n'
                      'output: result.msh2\nmesh:\n  fracture_size: 0.1\n')
    crop_override = ['--crop', '-.1', '.1', '-.1', '.1', '-.1', '.1'] if cli_override else []
    run = subprocess.run([sys.executable, str(APP/'convert.py'), '--config', str(config),
                          '--summary-only', *crop_override],
                         cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    summary = json.loads((tmp_path/'result.summary.json').read_text())
    assert 0 < summary['fractures'] < 8409
    radius = .1 if cli_override else .2
    assert summary['bounds'] == [[-radius, radius]]*3
    assert sum(f['count'] for f in summary['families']) == summary['fractures']


def test_cli_reverifies_saved_mesh_without_rebuilding_geometry(tmp_path):
    import numpy as np
    from dfn_data import NetworkData
    from mesh import generate_mesh, MeshConfig
    output = tmp_path/'existing.msh2'
    generate_mesh(NetworkData(np.array([[-.5, .5]]*3), ()), MeshConfig(bulk_size=.5), output)
    run = subprocess.run([sys.executable, str(APP/'convert.py'), '--output', str(output), '--verify-existing'],
                         cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads(output.with_suffix('.verification.json').read_text())
    assert result['status'] == 'verified'
    assert result['verification']['bulk_volume'] == 1


def test_cli_gamma_override_heals_existing_mesh_without_source(tmp_path):
    import numpy as np
    from dfn_data import NetworkData
    from mesh import generate_mesh, MeshConfig
    output = tmp_path/'raw.msh2'
    generate_mesh(NetworkData(np.array([[-.5, .5]]*3), ()), MeshConfig(), output)
    before = output.read_bytes()
    run = subprocess.run([sys.executable, str(APP/'convert.py'), '--output', str(output),
                          '--input', str(tmp_path/'absent'), '--heal-existing', '--gamma', '.002'],
                         cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    result = json.loads((tmp_path/'raw_healed.healing.json').read_text())
    assert result['gamma'] == .002
    assert result['status'] == 'completed'
    assert output.read_bytes() == before


@pytest.mark.parametrize('cli_override', [False, True])
def test_radius_config_and_cli_filter_scaled_bgem_export(tmp_path, cli_override):
    import numpy as np
    from dfn_data import load_network

    config = tmp_path/'settings.yaml'
    rmin, rmax = (9, 10) if cli_override else (.5, 1)
    config.write_text(f'input:\n  directory: {APP / "DFN_gen-out_v01_d004"}\n'
                      f'  rmin: {rmin}\n  rmax: {rmax}\n'
                      'output: filtered.msh2\ncoordinate_scale: 10\n')
    export = tmp_path/'fractures.npz'
    overrides = ['--rmin', '0.5', '--rmax', '1'] if cli_override else []
    run = subprocess.run([sys.executable, str(APP/'convert.py'), '--config', str(config),
                          *overrides, '--summary-only', '--bgem-output', str(export)],
                         cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    network = load_network(APP/'DFN_gen-out_v01_d004', coordinate_scale=10)
    selected = [f.source_rows[(f.arrays['fracture_req'] >= .5) &
                             (f.arrays['fracture_req'] <= 1.)] for f in network.families]
    summary = json.loads(run.stdout)
    assert summary['radius_filter'] == dict(rmin=.5, rmax=1., radius_field='fracture_req')
    assert summary['fractures'] == sum(map(len, selected))
    assert 0 < summary['fractures'] < len(network)
    with np.load(export) as arrays:
        for i, rows in enumerate(rows for rows in selected if len(rows)):
            np.testing.assert_array_equal(arrays[f'g{i}_source_rows'], rows)


def test_nested_input_selects_nondefault_domain_and_realization(tmp_path):
    import numpy as np
    import yaml
    from dfn_data import load_network

    family = load_network(APP/'DFN_gen-out_v01_d004').families[0].select([0])
    source = tmp_path/'source'
    source.mkdir()
    metadata = dict(domain_types=['cuboid', 'cuboid'],
                    domain_cuboid_xyzs=[[[-1, 1]]*3, [[-2, 2]]*3],
                    domain_frsets_names=[[], ['EW']], frsets_names=['EW'])
    (source/'DFN_gen-vars.yaml').write_text(yaml.safe_dump(metadata))
    selected = source/'ellipses_s00_d001_r002.npz'
    np.savez(selected, **family.arrays)
    selected.with_suffix('.yaml').write_text(yaml.safe_dump(dict(
        fracture_set_idx=0, domain_idx=1, realization=2, number_of_fractures=1,
        column_names=list(family.arrays))))
    config = tmp_path/'settings.yaml'
    config.write_text('input:\n  directory: source\n  domain_idx: 1\n  realization: 2\n'
                      'output: selected.msh2\n')
    run = subprocess.run([sys.executable, str(APP/'convert.py'), '--config', str(config),
                          '--summary-only'], cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    summary = json.loads(run.stdout)
    assert summary['fractures'] == 1
    assert summary['bounds'] == [[-2, 2]]*3
    assert summary['families'][0]['source'] == str(selected)


@pytest.mark.parametrize('geometry_tol', [None, 0., 1e-4])
def test_cli_geometry_tolerance_overrides_yaml_and_reports_bulk_metrics(tmp_path, geometry_tol):
    config = tmp_path/'settings.yaml'
    config.write_text(f'input:\n  directory: {APP / "DFN_gen-out_v01_d004"}\n'
                      '  rmin: 100\n  crop: [-.1, .1, -.1, .1, -.1, .1]\n'
                      'output: geometry.msh2\nmesh:\n  geometry_tol: 1e-3\n')
    overrides = [] if geometry_tol is None else ['--geometry-tol', str(geometry_tol)]
    geometry_tol = .001 if geometry_tol is None else geometry_tol
    run = subprocess.run([sys.executable, str(APP/'convert.py'), '--config', str(config),
                          *overrides],
                         cwd=tmp_path, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    report = json.loads((tmp_path/'geometry.report.json').read_text())
    assert report['config']['geometry_tol'] == geometry_tol
    assert report['geometry']['healing_applied'] == (geometry_tol > 0)
    for stage in ('before_healing', 'after_healing'):
        assert report['geometry'][stage]['area_perimeter']['min'] == pytest.approx(.05)
        assert report['geometry'][stage]['volume_surface_area']['max'] == pytest.approx(1/30)
    assert report['verification']['bulk_volume'] == pytest.approx(.008)
