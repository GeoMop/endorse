import pyvista as pv
"""
Local, gradient based optimization of boreholes and
position of measurements.

Approach:
1. From Total or first order sensitivity analysis indicators compute
   combined sensitivity index (e.g. minimum of parameter indices ).
2. Having combined index construct corresponding VTK Cell Data field F.
3. For given initial setup of borholes and sensor positions, determine perturbed setups
   maximizing the sensitivity index.
   - parameters: 8 well head positions X,Y; two angles a,b; 3 measurement positions; 7 * 8 = 56 in total
   - evalutaion of loss: Use pyvista mesh.extract_cells_along_line function.
   - optimization: optimize each borehole independently  
"""

# Assuming `mesh` is your PyVista mesh and `field_name` is the VTK Cell Data field
start_point = [x_start, y_start, z_start]
end_point = [x_end, y_end, z_end]

line = pv.Line(start_point, end_point)
cells_along_line = mesh.extract_cells_along_line(line)
average_value = mesh[field_name][cells_along_line].mean()