from OCC.Core.STEPControl import STEPControl_Writer, STEPControl_AsIs
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeShell, BRepBuilderAPI_MakeSolid
from OCC.Core.BRep import BRep_Builder
from OCC.Core.TopoDS import TopoDS_Shell
from OCC.Core.STL import stl_ReadFile
from OCC.Core.BRepTools import breptools_Write

def stl_to_brep(stl_filename, brep_filename):
    """
    Convert STL to BREP using pythonOCC-core.

    Parameters:
    - stl_filename: Path to the input STL file.
    - brep_filename: Path to the output BREP file.
    """
    # Step 1: Read STL file into a triangulated mesh
    triangulation = stl_ReadFile(stl_filename)
    if triangulation.IsNull():
        raise ValueError(f"Failed to read STL file: {stl_filename}")

    # Step 2: Build a shell from the triangulated mesh
    shell_maker = BRepBuilderAPI_MakeShell()
    shell_maker.Add(triangulation)

    if not shell_maker.IsDone():
        raise ValueError("Failed to create shell from triangulation.")
    
    shell = shell_maker.Shell()
    if shell.IsNull():
        raise ValueError("The generated shell is null.")

    # Step 3: Wrap the shell into a solid (if watertight)
    solid_maker = BRepBuilderAPI_MakeSolid(shell)
    if not solid_maker.IsDone():
        raise ValueError("Failed to create a solid from the shell.")
    solid = solid_maker.Solid()

    # Step 4: Write the solid to a BREP file
    breptools_Write(solid, brep_filename)
    print(f"BREP file successfully written to: {brep_filename}")

# Example usage
stl_filename = "simple_prism.stl"  # Replace with your STL file path
brep_filename = "simple_prism_converted.brep"  # Output BREP file path
stl_to_brep(stl_filename, brep_filename)
