"""Stream mesh geometry to ASCII GMSH 2.2 without rebuilding a Gmsh model."""
import json
from pathlib import Path


def write_msh22(mesh, output: Path):
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
    temporary.replace(output)
