from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np

from endorse.common import File
from endorse.mesh_class import load_mesh


def parse_args():
    parser = ArgumentParser(description="Compare tensor fields on two Gmsh meshes.")
    parser.add_argument("mesh_a", type=Path)
    parser.add_argument("mesh_b", type=Path)
    parser.add_argument("--field", default="conductivity_tn")
    parser.add_argument("--node-tol", type=float, default=1e-10)
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--worst", type=int, default=10, help="Number of worst elements to print.")
    return parser.parse_args()


def load(path: Path):
    return load_mesh(File(str(path)), heal_tol=None)


def node_mapping(mesh_a, mesh_b, tol):
    if mesh_a.nodes.shape != mesh_b.nodes.shape:
        raise ValueError(f"Different node array shape: {mesh_a.nodes.shape} vs {mesh_b.nodes.shape}")

    if np.allclose(mesh_a.nodes, mesh_b.nodes, atol=tol, rtol=0.0):
        return np.arange(len(mesh_a.nodes), dtype=int)

    order_a = np.lexsort(mesh_a.nodes.T[::-1])
    order_b = np.lexsort(mesh_b.nodes.T[::-1])
    sorted_a = mesh_a.nodes[order_a]
    sorted_b = mesh_b.nodes[order_b]
    if not np.allclose(sorted_a, sorted_b, atol=tol, rtol=0.0):
        diff = np.max(np.abs(sorted_a - sorted_b))
        raise ValueError(f"Node coordinates differ, max sorted difference: {diff}")

    map_a_to_b = np.empty(len(mesh_a.nodes), dtype=int)
    map_a_to_b[order_a] = order_b
    return map_a_to_b


def element_signature(element, node_map):
    return element.type, tuple(sorted(node_map[np.array(element.node_indices, dtype=int)]))


def element_mapping(mesh_a, mesh_b, node_map):
    if len(mesh_a.elements) != len(mesh_b.elements):
        raise ValueError(f"Different element count: {len(mesh_a.elements)} vs {len(mesh_b.elements)}")

    pools = defaultdict(list)
    for ib, el in enumerate(mesh_b.elements):
        pools[(el.type, tuple(sorted(el.node_indices)))].append(ib)

    map_a_to_b = np.empty(len(mesh_a.elements), dtype=int)
    for ia, el in enumerate(mesh_a.elements):
        signature = element_signature(el, node_map)
        try:
            map_a_to_b[ia] = pools[signature].pop()
        except IndexError as exc:
            raise ValueError(f"No matching element for A index {ia}, signature {signature}") from exc
        except KeyError as exc:
            raise ValueError(f"Missing signature for A index {ia}, signature {signature}") from exc

    leftovers = sum(len(v) for v in pools.values())
    if leftovers:
        raise ValueError(f"Unmatched elements in mesh B: {leftovers}")
    return map_a_to_b


def compare_field(field_a, field_b, rtol, atol):
    diff = field_a - field_b
    abs_diff = np.abs(diff)
    frob_per_element = np.linalg.norm(diff, axis=1)
    ref_per_element = np.linalg.norm(field_b, axis=1)
    rel_per_element = frob_per_element / np.maximum(ref_per_element, atol)
    return {
        "allclose": np.allclose(field_a, field_b, rtol=rtol, atol=atol),
        "l2": float(np.linalg.norm(diff)),
        "linf": float(np.max(abs_diff)),
        "mean_frob": float(np.mean(frob_per_element)),
        "max_frob": float(np.max(frob_per_element)),
        "max_rel": float(np.max(rel_per_element)),
        "frob_per_element": frob_per_element,
        "rel_per_element": rel_per_element,
    }


def print_summary(mesh_a, mesh_b, field_name, stats, worst_indices):
    print(f"mesh_a: {mesh_a.file.path}")
    print(f"mesh_b: {mesh_b.file.path}")
    print(f"field: {field_name}")
    print(f"nodes: {len(mesh_a.nodes)}")
    print(f"elements: {len(mesh_a.elements)}")
    print(f"allclose: {stats['allclose']}")
    print(f"global_l2: {stats['l2']}")
    print(f"global_linf: {stats['linf']}")
    print(f"mean_element_frobenius: {stats['mean_frob']}")
    print(f"max_element_frobenius: {stats['max_frob']}")
    print(f"max_element_relative: {stats['max_rel']}")
    print("worst_elements:")
    for ia in worst_indices:
        print(
            f"  a_idx={ia}, "
            f"frob={stats['frob_per_element'][ia]}, "
            f"rel={stats['rel_per_element'][ia]}"
        )


def main():
    args = parse_args()
    mesh_a = load(args.mesh_a)
    mesh_b = load(args.mesh_b)

    a_to_b_nodes = node_mapping(mesh_a, mesh_b, args.node_tol)
    a_to_b_elements = element_mapping(mesh_a, mesh_b, a_to_b_nodes)

    field_a = mesh_a.get_static_p0_values(args.field)
    field_b = mesh_b.get_static_p0_values(args.field)[a_to_b_elements]
    if field_a.shape != field_b.shape:
        raise ValueError(f"Different field shape: {field_a.shape} vs {field_b.shape}")

    stats = compare_field(field_a, field_b, args.rtol, args.atol)
    worst = np.argsort(stats["frob_per_element"])[-args.worst:][::-1]
    print_summary(mesh_a, mesh_b, args.field, stats, worst)
    raise SystemExit(0 if stats["allclose"] else 1)


if __name__ == "__main__":
    main()
