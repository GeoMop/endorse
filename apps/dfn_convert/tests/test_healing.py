from pathlib import Path

import gmsh
import pytest

from healing import heal_mesh_file


def write_sliver(path):
    path.write_text('''$MeshFormat
2.2 0 8
$EndMeshFormat
$PhysicalNames
1
3 1 "bulk"
$EndPhysicalNames
$Nodes
5
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
5 0.3 0.3 0.000001
$EndNodes
$Elements
2
1 4 2 1 1 1 2 3 4
2 4 2 1 1 1 2 3 5
$EndElements
''')


def test_gamma_drives_real_healmesh_and_preserves_raw_mesh(tmp_path):
    raw = tmp_path/'sliver.msh2'
    write_sliver(raw)
    before = raw.read_bytes()
    low = heal_mesh_file(raw, gamma=1e-16)
    assert low['quality_before']['below_gamma'] == 0
    high = heal_mesh_file(raw, gamma=.01)
    assert high['gamma'] == .01
    assert high['quality_before']['below_gamma'] > 0
    assert high['quality_after']['below_gamma'] < high['quality_before']['below_gamma']
    assert raw.read_bytes() == before
    healed = Path(high['output'])
    assert healed.read_text().splitlines()[:2] == ['$MeshFormat', '2.2 0 8']
    assert not gmsh.isInitialized()


@pytest.mark.parametrize('gamma', [-.01, 0, 1.1, float('nan')])
def test_invalid_gamma_fails_before_reading_mesh(tmp_path, gamma):
    with pytest.raises(ValueError, match='gamma'):
        heal_mesh_file(tmp_path/'missing.msh2', gamma)


def test_read_failure_is_reported_and_gmsh_is_closed(tmp_path):
    import json
    with pytest.raises(FileNotFoundError):
        heal_mesh_file(tmp_path/'missing.msh2', gamma=.01)
    assert not gmsh.isInitialized()
    report = json.loads((tmp_path/'missing_healed.healing.json').read_text())
    assert report['status'] == 'failed'


def test_healed_output_with_detached_surface_is_rejected(tmp_path):
    import json
    raw = tmp_path/'detached.msh2'
    write_sliver(raw)
    raw.write_text(raw.read_text().replace('1\n3 1 "bulk"', '2\n3 1 "bulk"\n2 2 "EW"')
                   .replace('$Elements\n2\n', '$Elements\n3\n')
                   .replace('$EndElements', '3 2 2 2 2 2 4 5\n$EndElements'))
    with pytest.raises(ValueError, match='surface'):
        heal_mesh_file(raw, gamma=1e-16)
    result = json.loads((tmp_path/'detached_healed.healing.json').read_text())
    assert result['status'] == 'failed'
    assert result['structural_check']['detached_surface_triangles'] == 1


@pytest.mark.parametrize('coordinates', ['nan', 'overflow'])
def test_nonfinite_healed_geometry_is_rejected(tmp_path, coordinates):
    from healing import _audit_output
    path = tmp_path/'invalid.msh2'
    write_sliver(path)
    text = path.read_text()
    if coordinates == 'nan':
        text = text.replace('1 0 0 0', '1 nan 0 0')
    else:
        text = text.replace('2 1 0 0', '2 1e200 0 0').replace('3 0 1 0', '3 0 1e200 0').replace('4 0 0 1', '4 0 0 1e200')
    path.write_text(text)
    result = _audit_output(path)
    assert not result['passed']
    assert result['bulk_volume'] is None
    assert not gmsh.isInitialized()
