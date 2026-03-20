from math import *
import scipy.optimize
from attrs import define
import numpy as np
import pyvista as pv
from endorse import common
from chodby_inv.hm_model.boreholes import Boreholes
import chodby_inv.input_data as input_data
import bgem.gmsh.gmsh

@define
class LineObj:
    # name: str
    start_pt: np.ndarray(3,)
    direction: np.ndarray(3,)
    length: float

def create_line_objs(cfg:'dotdict', bhs: Boreholes) -> dict[str, LineObj]:
    line_objs = {}
    for obj_name,obj_data in cfg.items():
        bh_idx = bhs.bh_index_from_name(obj_name)
        if bh_idx > 0:
            line_objs[obj_name] = LineObj(start_pt=bhs.bh_start(bh_idx),
                                          direction=bhs.bh_direction(bh_idx),
                                          length=bhs.bh_length(bh_idx))
        else:
            line_objs[obj_name] = LineObj(**obj_data)
    return line_objs

def generate_fractures(cfg:'dotdict', line_objs: dict[str, LineObj], l5_azimuth):

    def compute_normal(azimuth, inclination, l5_azimuth):
        x = cos((l5_azimuth + 90 - azimuth) * pi / 180)
        y = sin((l5_azimuth + 90 - azimuth) * pi / 180)
        z = sin(inclination * pi / 180)
        return np.array([x, y, z])

    def fit_plane_with_normal_constraint(points, n_given, point_weights=None, n_weight=1.0, ns_weight=1.0):
        """
        points: (N,3) np.array - prescribed plane points
        n_given: (3,) np.array - prescribed normal direction
        point_weights: (N,) np.array - weights of points
        n_weight: weight of n_given
        ns_weight: weight of constraint |n|=1
        """
        points = np.asarray(points)
        n_given = np.asarray(n_given)
        n_given = n_given / np.linalg.norm(n_given)
        N = points.shape[0]

        # diagonal matrix of weights
        if point_weights is None:
            point_weights = np.ones(N)
        else:
            point_weights = np.asarray(point_weights)

        def f(x):
            x = np.asarray(x)
            n = x[:3]
            d = x[3]
            val = 0
            for i in range(N):
                val += point_weights[i]*(np.dot(points[i,:],n) - d)**2
            val += n_weight*np.dot(n-n_given,n-n_given)
            val += ns_weight*(np.dot(n,n)-1)**2
            return val

        # solution
        x0 = np.concatenate( (n_given, np.array([0])) )
        result = scipy.optimize.minimize(f, x0, options={'maxiter':1000})

        # split into parts
        x = np.asarray(result.x)
        n = x[:3]
        d = x[3]

        return n, d, result

    fractures = {}
    for fr_name,fr_data in cfg.items():
        pts = []
        wts = []
        for pt in fr_data.points:
            start = np.array(line_objs[pt.line_object].start_pt)
            dir = np.array(line_objs[pt.line_object].direction)
            pts.append(start + dir*pt.position)
            wts.append(pt.weight)
        normal = compute_normal(fr_data.normal.azimuth, fr_data.normal.inclination, l5_azimuth)
        n,d,r = fit_plane_with_normal_constraint(pts, normal,
                                                  point_weights=wts,
                                                  n_weight=fr_data.weight_normal,
                                                  ns_weight=fr_data.weight_normal_size)
        print(f'{fr_name}: normal={n} (norm {np.linalg.norm(n)}) d={d} residual={r.fun} nit={r.nit} msg="{r.message}"')
        fractures[fr_name] = [ n,d ]

    return fractures


def create_planes_vtk(planes, size=50.0, filename="generated_fracture.vtm"):
    """
    Create a square plane with given 'size', normal 'n' and offset 'd', and saves it to VTK format.

    planes: dict with structure "name":(normal,point_in_plane)
    size: square side length
    filename: output file name
    """
    mb = pv.MultiBlock()

    for plane_name,plane in planes.items():
        n = np.asarray(plane[0])
        n = n / np.linalg.norm(n)

        # point in plane
        d = plane[1]
        p0 = d * n  # provided n has unit size

        # tangent vectors
        if abs(n[0]) < abs(n[1]):
            v = np.array([1, 0, 0])
        else:
            v = np.array([0, 1, 0])

        t1 = np.cross(n, v)
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        t2 /= np.linalg.norm(t2)

        # square corners
        half = size / 2
        corners = np.array([
            p0 + half * (t1 + t2),
            p0 + half * (t1 - t2),
            p0 + half * (-t1 - t2),
            p0 + half * (-t1 + t2)
        ])

        # generate PolyData for VTK
        poly = pv.PolyData()
        poly.points = corners
        poly.faces = np.hstack([[4, 0, 1, 2, 3]])  # one square
        mb[plane_name] = poly

    # save to file
    mb.save(filename)
    print(f"Plane saved to {filename}")


def create_planes_gmsh(factory:bgem.gmsh.gmsh.GeometryOCC, planes, size=50.0):
    """
    Create a square plane with given 'size', normal 'n' and offset 'd', and return a list of GMSH objects.

    factory: GMSH factory
    planes: dict with structure "name":(normal,point_in_plane)
    size: square side length
    """

    fracs = {}

    for plane_name,plane in planes.items():
        n = np.asarray(plane[0])
        n = n / np.linalg.norm(n)

        # point in plane
        d = plane[1]
        p0 = d * n  # provided n has unit length

        fracs[plane_name] = factory.disc_discrete(size, p0, axis=n, n_points=12)

    return fracs



if __name__ == "__main__":
    cfg = common.load_config(input_data.l5_mesh_cfg_yaml)
    bhs = Boreholes(cfg.boreholes)
    line_objs = create_line_objs(cfg.line_objects, bhs)
    fracs = generate_fractures(cfg.generated_fractures, line_objs, cfg.boreholes.geometry.l5_azimuth)
    create_planes_vtk(fracs)