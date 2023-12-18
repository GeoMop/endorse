from typing import  *
import glob
import attrs
from functools import cached_property
import itertools
import numpy as np
import pyvista as pv
from vtk import vtkCellLocator
"""
TODO:
1. Test problem genetic algorithm on cluster.
2. Single individual : N borehole ids, 4*N plug position ids
   ZK30 - N=6, 6 + 24 = 30 ints
3. validation of individual - check borehole interactions - for every pair compute transwersal, it must be > 1m 
4. Mutation : change with some probablility in every borehole group , change the plug position by certain step (-3 up to +3) 
5. Mating : one of borehole setups in every independent group; theta convex interpolation between plug positions, should preserve ordering
6. Evalutaion : 
"""

@attrs.define
class WHZone:
    """
    Defines XYZ range of borehole start points
    + range of horizontal, vertical angles in terms of target points at distance 10m
    """
    box: np.ndarray         # shape (2,3) [min/max, XYZ]
#    directions: np.ndarray  # shape (2,2)  [min/max, YZ]


@attrs.define(slots=False)
class BoreholeSet:
    """
    Class for creating a set of boreholes around single lateral using its coordinate system:
    - meaning of axis is same, but X and Y have opposed sign
    - origin is at the center of avoidance cylinder at intersection of L5 and lateral central lines.
    """
    transform_matrix: np.ndarray    # Mapping from lateral system to main system
    transform_shift: np.array       # position of the lateral system origin
    y_angles: np.ndarray            # y boreholes angles to consider
    z_angles: np.ndarray            # z borehole angles to consider
    wh_pos_step: Tuple[float, float, float] # xyz wellhead position step

    avoid_cylinder: Tuple[float, float, float]     # Boreholes can not insterset this cylinder (r, length_min, length_max) from origin in X direction
    active_cylinder: Tuple[float, float, float]    # Boreholes must intersect this cylinder (r, length_min, length_max) from origin in X direction
    wh_zones: List[WHZone]
    _y_angle_range = (-80, 80)      # achive 2m from 10m distance
    _z_angle_range = (-60, 60)

    def transform(self, point):
        return self.transform_matrix @ point + self.transform_shift

    @cached_property
    def cylinder_line(self):
        return np.array([0,0,0,1,0,0])

    @staticmethod
    def _angle_array(lst):
        if lst[0] == 0.0:
            # Make symmetric.
            lst = [-x for x in lst[1:]] + lst
        return np.array(lst)

    @cached_property
    def _axis_angles(self):
        return [self._angle_array(self.y_angles), self._angle_array(self.z_angles)]

    def axis_angles(self, axis):
        return self._axis_angles[axis]

    @staticmethod
    def direction(y_phi, z_phi):
        y_phi = y_phi / 180 * np.pi
        z_phi = z_phi / 180 * np.pi
        sy, cy = np.sin(y_phi), np.cos(y_phi)
        sz, cz = np.sin(z_phi), np.cos(z_phi)
        return np.array([cy * cz, sy * cz, sz])

    @property
    def n_boreholes(self):
        return len(self.bh_list)

    @property
    def n_y_angles(self):
        return len(self.axis_angles(0))

    @property
    def n_z_angles(self):
        return len(self.axis_angles(1))


    @staticmethod
    def linspace(a, b, step):
        a, b =  (a, b) if a < b else (b, a)
        n = int((b - a) / step)
        if n < 1:
            return np.array([(a+b) / 2])
        else:
            return np.linspace(a, b, n + 1)

    @property
    def angles_table(self):
        angle_tab, bh_list = self._build_boreholes
        return angle_tab

    @property
    def bh_list(self):
        angle_tab, bh_list = self._build_boreholes
        return bh_list


    def direction_lookup(self, i, j):
        return [self.bh_list[i] for i in self.angles_table[i][j]]

    def draw_angles(self, n):
        return np.stack(
             [np.random.randint(self.n_y_angles, size=n),
                   np.random.randint(self.n_z_angles, size=n)],
                   axis=1)

    @cached_property
    def _build_boreholes(self):
        ny, nz = len(self.axis_angles(0)), len(self.axis_angles(1))
        bh_dir_table = [[[] for j in range(nz)] for i in range(ny)]
        bh_list = []
        for zone in self.wh_zones:
            ranges = [self.linspace(*zone.box[:, idim], self.wh_pos_step[idim]) for idim in [0, 1, 2]]
            xyz_range = itertools.product(*ranges)
            for pos in xyz_range:
                for i, j, bh in self._build_position(pos):
                    bh_dir_table[i][j].append(len(bh_list))
                    bh_list.append(bh)

        return bh_dir_table, bh_list


    def _build_position(self, pos):
        yz_phi_range = itertools.product(enumerate(self._axis_angles[0]), enumerate(self._axis_angles[1]))
        for (i_phi, y_phi), (j_phi, z_phi) in yz_phi_range:
            bh = self._make_borehole(pos, y_phi, z_phi)
            if bh is not None:
                yield (i_phi, j_phi, bh)

    def _make_borehole(self, pos, y_phi, z_phi):
        dir = self.direction(y_phi, z_phi)
        length, cyl_t, bh_t, cyl_point, bh_point = self.transversal_params(self.cylinder_line, np.array([*pos, *dir]))
        bh_dir = bh_point - pos
        # print(f"({y_phi}, {z_phi}) dir: {dir} {bh_t} bh: {bh_dir} dot: {np.dot(dir, bh_dir)}")
        r, l0, l1 = self.avoid_cylinder
        if length < r and l0 < cyl_t < l1:
            return None
        r, l0, l1 = self.active_cylinder
        if length < r and l0 < cyl_t < l1:
            return np.array([*pos, *bh_dir])
        return None

        #return np.array([*pos, *bh_dir])

    @staticmethod
    def transversal_lengths(line1, lines):
        """
        Calculate the length of the transversal between a single line and multiple lines using vectorized operations.

        Parameters:
        line1: Numpy array representing the first line (point and direction vector)
        lines: Numpy array of shape (N, 6), each row representing a line (point and direction vector)

        Returns:
        Numpy array of lengths of the transversals
        """
        # Extract point and direction vector from line1
        a1, d1 = line1[wellhead], line1[direction]

        # Extract points and direction vectors from lines
        a2s, d2s = lines[:, wellhead], lines[:, direction]

        # Compute cross products for all lines in a vectorized manner
        cross_products = np.cross(d1[None, :], d2s)

        # Calculate lengths of the transversals
        differences = a2s - a1
        lengths = np.abs(differences[:,None, :] @ cross_products[:, :, None])[:,0,0] / np.linalg.norm(cross_products, axis=1)

        return lengths

    @staticmethod
    def transversal_params(line1, line2):
        a1, d1 = line1[:3], line1[3:]
        a2, d2 = line2[:3], line2[3:]

        # Build the orthogonal coordinate system
        ex = np.cross(d1, d2)
        norm_ex = np.linalg.norm(ex)
        if np.isnan(norm_ex):
            return np.inf, np.inf, np.inf, 10 * d1 , 10 * d2
        ex_normalized = ex / norm_ex
        ey = d1 / np.linalg.norm(d1)
        ez = np.cross(d1, ex)
        ez_normalized = ez / np.linalg.norm(ez)

        diff = a2 - a1
        # Project (a2 - a1) onto ez to find t
        t = np.dot(diff, ez_normalized)
        point_2 = a2 + t * d2

        # Calculate s
        s = np.dot(point_2 - a1, ey)
        point_1 = a1 + s * d1

        length = np.abs(np.dot(diff, ex_normalized))
        return length, s, t, point_1, point_2


def BHS_zk_30():
    return BoreholeSet(
        transform_matrix=np.eye(3),
        transform_shift=np.array([0, -5, 1.8]),
        #dir_angle_step = (5, 5),     # degree
        #wh_pos_step = (1, 1, 0.5),
        y_angles = np.linspace(-80, 80, 9),     # degree
        z_angles = [0, 10, 25, 45, 70],
        wh_pos_step = (3, 3, 1),
        avoid_cylinder = (3, 0, 12),        # r, x0, x1
        active_cylinder = (20, 3, 30),      # r, x0, x1
        wh_zones=[
            WHZone(
                box=np.array([
                    [-1, -5, -0.4],   # min
                    [+1, -20, 0.1]   # max
                ]),
            ),
            WHZone(
                box=np.array([
                    [-1, 15, -0.4],  # min
                    [+1, 25, 0.1]  # max
                ]),
                # directions=np.array([
                #     [0, -20],
                #     [25, +20]
                # ])
            )

        ]

    )





# Line has 9 coordinates, 3 points:
# wellhead, direction vector to the endpoint, s - transversal intersection with respect to tunnel center
# line is given by:
# "wellhead" which is in fact start of the drilling
# transversal intersection, which is parametrized by position in the lateral tunnel and angle in XZ plane
wellhead = slice(0,3)
direction = slice(3,6)
transversal_param = slice(6, 6)
def line_end(line):
    return line[..., wellhead] + line[..., direction]

def transversal_point(line):
    return line[..., wellhead] + line[..., transversal_param][0] * line[..., direction]


"""
Single borehole given by:
position on L5 vertical axis plane, small vertical range, large horizontal range, target point

- wellheads generated uniformly over given ranges
- for each well head we generate range of direction vectors
- eliminate boreholes interacting with cylinder
Jedna konfigurace vrtů:
restrictions:
- 2 boreholes for ZK40, parallel under, 6 boreholes from ZK30, 2 parallel under, 4 generic 2 upper
=> directions to upper and lower, group boreholes according to direction, able to get wellheads for given direction
? how to prescribe restrictions to genetic algorithm

Separate optimization of:
 
 - ZK40 2 lower
 - ZK30 2 lower parallel, 2 upper generic, 2 generic 
 - ZK40 2 upper 
 In principle I'm takeing given number of boreholes from defined borehole subsets.
 Muting and mating within these subsets provides resonable results. 
 Posterior check of borehole interactions necessary, we iterate until we get viable ofspring.
  
ZK40: upper/lower, index of direction, 2 indeces from available 
"""




def get_time_field(file_pattern, field_name):
    """
    Read files given by pattern, separate into numpy array and mesh.

    Return: field array, mesh
    - field array of shape (n_times, n_elements)
    - mesh without data fields
    """
    time_slices = []
    files = glob.glob(file_pattern)
    if not files:
        raise IOError(file_pattern)
    for filepath in files:
        mesh = pv.read(filepath)
        time_slices.append(mesh.cell_data.get_array(field_name))
    mesh.clear_data()
    return np.array(time_slices), mesh

def line_points(lines, n_points):
    """

    :param lines: (n_lines, 6) [*point, *direction]
    :param n_points:
    :return: (n_lines, n_points, 3)
    """
    origin = lines[:, :3]
    direction = lines[:, 3:]
    params = np.linspace(0, 1, n_points)
    return direction[:, None, :] * params[:,None] + origin[:, None, :]

def interpolation_slow(mesh, lines, n_points):
    """
    Construct element_id matrix of shape (n_boreholes, n_points).

    Put n_points on every line
    :param mesh:
    :param lines/
    :param n_points:
    :return:
    """
    points = line_points(lines, n_points)
    cell_locator = vtkCellLocator()
    cell_locator.SetDataSet(mesh)
    cell_locator.BuildLocator()

    interpolation_ids = np.empty([*points.shape[:2]])
    for i_line in range(len(lines)):
        for i_pt in range(n_points):
            interpolation_ids[i_line, i_pt] = cell_locator.FindCell(points[i_line, i_pt])
    return interpolation_ids

def get_values_on_lines(mesh, values, id_matrix):
    """
    :param mesh:
    :param values: shape(n_samples, n_times, n_cells)
    :param id_matrix:
    """
    result = values[:, :, id_matrix]
    result = np.moveaxis(result, 2, 0)
    return result


"""
Úloha jak z daného bodu generovat vrty, které se vyhnou tunelu
- tunel/rozrážka zjednodušen na válec, kterému se potřebujeme vyhnout
- vrt daný příčkou mimoběžek (vrt vs. osa tunelu), 
  ta daná parametrem (st) bodu  (P) na ose tunelu
  a úhlem (phi) (směr vzhůru je phi=0) + pravidlo pravé ruky s palcem ve směru tunelu
- příčka nakonec dána jednotkovou normálou N a bodem na ose tunelu P
- rovnice přímky vrtu zhlaví PV, směrnice T:

  PV .. wellhead (given)
  Tv .. direction of well
  Pt .. center of (lateral) tunnel on the border of the L5 gallery (given), like (+/- 2, -/+ 5, 2) 
  Tt .. unit direction vector of tunnel (given) (+/- 1, 0, 0)

  We consider transversal of skew lines: the tunnel center line and the well
  It intersects and is perpendicular to both lines. 
  We consider the transversl given by its angle phi in YZ plane and by
  its intersection with the tunnel center line:
  N = (0, sin(phi), cos(phi))
  P = st * Tt + Pt # transwersal equation
  phi=0 .... upwards, e_z  
  phi ... jeden z parametrů vrtu


  rovnice vrtu:
  X = v * Tv  + PV

  1. rovina obsahující příšku a PV
  n1 = (P - PV) - (P-PV).N N  
  n1 = n1 / |n1|

  n2 = cross(N, n1)
  normálová rovnice roviny:
  X . n1 = PV . n1 

  (T . t + PV) . n1 = PV . n1
  T. t . n1 = 0

  2. z pruseciku roviny kolme na N a prochazejici PV:
  rovina: X . N = PV . N
  pricka: X = N . s + P
  s + P . N = PV . N
  s = (PV - P) . N

  přímka vrtu:
  prochází skrze PV a (PV-P)@N * N * s + P
  X = PV + v * ((PV-P)@N * N * s - (PV-P))

  We want s > r(st). 
  So how P and N exactly looks like:

  Lp = abs(PVy - yt)    # vzdálenost zhlaví od pozice rozrážky 
  Lt .. délka tunelu

  přímka osy tunelu ve směru X:
  Tt = (+/-1, 0, 0)
  Pt = (xt, zy, zt)  ; fixed, xt = 2 (half width of the main tunnel), 
  P = Tt . st + Pt, 0 < st < Lt 

  st .. parameter of the tunnel center line
  r(st) = min(Rt, Lp + st * (Rt - Lp) / Lt)    


  So the plusability criteria is:
  (PV - Tt.st - Pt) . (0, sin(phi), cos(phi))  > Rt
  or 

  (PV - Tt.st - Pt) . (0, sin(phi), cos(phi))  > Lp + st * (Rt - Lp) / Lt
  (PV - Pt).N - Lp > [(Tt . N) + (Rt-Lp) / Lt ] * st 

  Tt . N = 0 =>
  (PV - Pt).N - Lp > (Rt-Lp) / Lt  * st
  st <  [(PV - Pt).N - Lp] * Lt / (Rt - Lp) 

"""