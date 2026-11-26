"""Stream mesh geometry to ASCII GMSH 2.2 without rebuilding a Gmsh model."""
import json
from pathlib import Path


def write_fracture_radius(stream, count, groups):
    """Write a scalar MSH 2.2 ElementData section from groups of element IDs."""
    stream.write(f'$ElementData\n1\n"fracture_radius"\n1\n0\n3\n0\n1\n{count}\n')
    for ids, radius in groups:
        value = format(radius, '.17g')
        stream.writelines(f'{eid} {value}\n' for eid in ids)
    stream.write('$EndElementData\n')


def fracture_radius_by_entity(mesh):
    """Recover radii by element type and CAD tag, which HealMesh copies."""
    field = getattr(mesh, 'element_data', {}).get('fracture_radius')
    if field is None:
        return None
    data = field[0]
    radii = {}
    for eid, value in zip(data.tags, data.values):
        kind, tags, _ = mesh.elements[int(eid)]
        radii[kind, tags[1]] = float(value[0])
    return radii


def write_msh22(mesh, output: Path, radii=None):
    """Write HealMesh/GmshIO geometry, retaining IDs, tags and physical names."""
    output = Path(output)
    temporary = output.with_name(output.name+'.writing')
    with temporary.open('w', buffering=1024*1024) as stream:
        stream.write('$MeshFormat\n2.2 0 8\n$EndMeshFormat\n')
        stream.write(f'$PhysicalNames\n{len(mesh.physical)}\n')
        for name, (tag, dim) in mesh.physical.items():
            stream.write(f'{dim} {tag} {json.dumps(name, ensure_ascii=False)}\n')
        stream.write('$EndPhysicalNames\n')
        stream.write(f'$Nodes\n{len(mesh.nodes)}\n')
        for node, (x, y, z) in mesh.nodes.items():
            stream.write(f'{node} {x:.17g} {y:.17g} {z:.17g}\n')
        stream.write('$EndNodes\n')
        stream.write(f'$Elements\n{len(mesh.elements)}\n')
        for element, (kind, tags, nodes) in mesh.elements.items():
            values = [element, kind, len(tags), *tags, *nodes]
            stream.write(' '.join(map(str, values))+'\n')
        stream.write('$EndElements\n')
        if radii is not None:
            groups = (((eid,), radii[kind, tags[1]])
                      for eid, (kind, tags, _) in mesh.elements.items())
            write_fracture_radius(stream, len(mesh.elements), groups)
    temporary.replace(output)
