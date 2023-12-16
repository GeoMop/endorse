import pyvista as pv
import numpy as np
import glob

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
# def create_scene(cfg_geometry):
#     cfg = cfg_geometry
#     # Create a plotting object
#     plotter = pv.Plotter()
#
#     # Create a horizontal cylinder
#     cylinder = pv.Cylinder(center=(5, 0, 0), direction=(1, 0, 0), radius=3, height=10)
#     plotter.add_mesh(cylinder, color='blue')
#     plotter.add_axes()
#     plotter.show_bounds(grid='front', all_edges=True)
#     return plotter
#
# def add_line_segment(plotter, point1, point2, color='red', line_width=2):
#     # Create a line between the two points
#     line = pv.Line(point1, point2)
#
#     # Add the line to the plotter
#     plotter.add_mesh(line, color=color, line_width=line_width)
#
# # Example usage
#
#
# plotter = create_scene()
#
# add_line_segment(plotter, [0, 0, 0], [5, 5, 5])
# plotter.show()


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



def length_of_transversals(line1, lines):
    """
    Calculate the length of the transversal between a single line and multiple lines using vectorized operations.

    Parameters:
    line1: Numpy array representing the first line (point and direction vector)
    lines: Numpy array of shape (N, 6), each row representing a line (point and direction vector)

    Returns:
    Numpy array of lengths of the transversals
    """
    # Extract point and direction vector from line1
    a1, d1 = line1[:3], line1[3:]

    # Extract points and direction vectors from lines
    a2s, d2s = lines[:, :3], lines[:, 3:]

    # Compute cross products for all lines in a vectorized manner
    cross_products = np.cross(d1[None, :], d2s)

    # Calculate lengths of the transversals
    differences = a2s - a1
    lengths = np.abs(differences[:,None, :] @ cross_products[:, :, None])[:,0,0] / np.linalg.norm(cross_products, axis=1)

    return lengths

def get_time_field(file_pattern, field_name):
    """
    Return: field array, geometry
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

def get_values_on_lines(mesh, values, lines, n_points):
    mesh.cell_data['pressure'] = values
    # create
    point_cloud = pv.PolyData(line_points(lines, n_points))
    mesh_with_point_data = mesh.cell_data_to_point_data()
    sampled_data = mesh_with_point_data.sample(point_cloud)
    sampled_values = sampled_data.point_arrays['pressure']



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