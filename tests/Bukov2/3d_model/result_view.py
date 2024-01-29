# state file generated using paraview version 5.10.1

# uncomment the following three lines to ensure this script works in future versions
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [858, 793]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.0, 0.0, 2.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-110.69386231544274, -48.46675844837146, 93.75874995133924]
renderView1.CameraFocalPoint = [67.27778547764042, 49.42134957160335, -77.05940757993615]
renderView1.CameraViewUp = [0.6376211521762147, 0.15685786185767242, 0.754211427565849]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 68.68943944928968
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(858, 793)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
flowpvd = PVDReader(registrationName='flow.pvd', FileName='/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_pressure_results/output_h100_init/flow.pvd')
flowpvd.CellArrays = ['region_id', 'pressure_p0', 'velocity_p0', 'conductivity']

# create a new 'Calculator'
calculator1 = Calculator(registrationName='limited_pressure', Input=flowpvd)
calculator1.AttributeType = 'Cell Data'
calculator1.ResultArrayName = 'pressure_lim'
calculator1.Function = 'max(pressure_p0, 0.3)'

# create a new 'Annotate Time Filter'
annotateTimeFilter1 = AnnotateTimeFilter(registrationName='AnnotateTimeFilter1', Input=calculator1)

# create a new 'Slice'
slicezk30_2 = Slice(registrationName='slice zk30_2', Input=calculator1)
slicezk30_2.SliceType = 'Plane'
slicezk30_2.HyperTreeGridSlicer = 'Plane'
slicezk30_2.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slicezk30_2.SliceType.Origin = [4.0, 0.0, 0.0]

# create a new 'Slice'
slicezk30_4 = Slice(registrationName='slice zk30_4', Input=calculator1)
slicezk30_4.SliceType = 'Plane'
slicezk30_4.HyperTreeGridSlicer = 'Plane'
slicezk30_4.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slicezk30_4.SliceType.Origin = [6.0, 0.0, 0.0]

# create a new 'Slice'
slicezk30_6 = Slice(registrationName='slice zk30_6', Input=calculator1)
slicezk30_6.SliceType = 'Plane'
slicezk30_6.HyperTreeGridSlicer = 'Plane'
slicezk30_6.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slicezk30_6.SliceType.Origin = [8.0, 0.0, 0.0]

# create a new 'Slice'
slicezk30_8 = Slice(registrationName='slice zk30_8', Input=calculator1)
slicezk30_8.SliceType = 'Plane'
slicezk30_8.HyperTreeGridSlicer = 'Plane'
slicezk30_8.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slicezk30_8.SliceType.Origin = [10.0, 0.0, 0.0]

# create a new 'Slice'
slicezk30_10 = Slice(registrationName='slice zk30_10', Input=calculator1)
slicezk30_10.SliceType = 'Plane'
slicezk30_10.HyperTreeGridSlicer = 'Plane'
slicezk30_10.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slicezk30_10.SliceType.Origin = [12.0, 0.0, 0.0]



# create a new 'Slice'
slicezk40_2 = Slice(registrationName='slice zk40_2', Input=calculator1)
slicezk40_2.SliceType = 'Plane'
slicezk40_2.HyperTreeGridSlicer = 'Plane'
slicezk40_2.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slicezk40_2.SliceType.Origin = [-4.0, 0.0, 0.0]

# create a new 'Slice'
slicezk40_4 = Slice(registrationName='slice zk40_4', Input=calculator1)
slicezk40_4.SliceType = 'Plane'
slicezk40_4.HyperTreeGridSlicer = 'Plane'
slicezk40_4.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slicezk40_4.SliceType.Origin = [-6.0, 0.0, 0.0]

# create a new 'Slice'
slicezk40_6 = Slice(registrationName='slice zk40_6', Input=calculator1)
slicezk40_6.SliceType = 'Plane'
slicezk40_6.HyperTreeGridSlicer = 'Plane'
slicezk40_6.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slicezk40_6.SliceType.Origin = [-8.0, 0.0, 0.0]

# create a new 'Slice'
slicezk40_8 = Slice(registrationName='slice zk40_8', Input=calculator1)
slicezk40_8.SliceType = 'Plane'
slicezk40_8.HyperTreeGridSlicer = 'Plane'
slicezk40_8.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slicezk40_8.SliceType.Origin = [-10.0, 0.0, 0.0]

# create a new 'Slice'
slicezk40_10 = Slice(registrationName='slice zk40_10', Input=calculator1)
slicezk40_10.SliceType = 'Plane'
slicezk40_10.HyperTreeGridSlicer = 'Plane'
slicezk40_10.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slicezk40_10.SliceType.Origin = [-12.0, 0.0, 0.0]



# create a new 'Clip'
cliphorizontal = Clip(registrationName='clip-horizontal', Input=calculator1)
cliphorizontal.ClipType = 'Plane'
cliphorizontal.HyperTreeGridClipper = 'Plane'
cliphorizontal.Scalars = ['CELLS', 'conductivity']
cliphorizontal.Value = 841334227.2506754

# init the 'Plane' selected for 'ClipType'
cliphorizontal.ClipType.Origin = [0.0, 0.0, 2.0]
cliphorizontal.ClipType.Normal = [0.0, 0.0, 1.0]

# create a new 'Clip'
test_subblock = Clip(registrationName='test_subblock', Input=calculator1)
test_subblock.ClipType = 'Box'
test_subblock.HyperTreeGridClipper = 'Plane'
test_subblock.Scalars = ['CELLS', 'conductivity']
test_subblock.Value = 841334227.2506754

# init the 'Box' selected for 'ClipType'
test_subblock.ClipType.Position = [-10.0, -10.0, -2.0]
test_subblock.ClipType.Length = [20.0, 20.0, 10.0]


#===========================================
# Manually turn off Show Plane/Clip/Box/...

def hide_widget(item_name):
    item = FindSource(item_name)
    
    renderView1 = GetActiveViewOrCreate('RenderView')
    #prop = GetDisplayProperties(item, view=renderView1)
    #vis = prop.Visibility
    #print(item_name, vis)
    

    if item is None:
        raise AttributeError("Unknown item: ", item_name)
    SetActiveSource(item)
    #item_type = type(item).__name__.split('.')[-1]
    try:
        Show3DWidgets(proxy=item.ClipType)
        Hide3DWidgets(proxy=item.ClipType)
    except AttributeError as e:
        pass
    try:
        Show3DWidgets(proxy=item.SliceType)
        Hide3DWidgets(proxy=item.SliceType)
    except AttributeError as e:
        pass
    Hide(item, renderView1)
    #prop.Visibility = vis
    #renderView1 = GetActiveViewOrCreate('RenderView')
    #prop = GetDisplayProperties(item, view=renderView1)

def hide_all_widgets():
    hide_widget('test_subblock')
    hide_widget('clip-horizontal')

    positions = [2, 4, 6, 8, 10]
    for i in positions:
        hide_widget(f"slice zk30_{i}")
        hide_widget(f"slice zk40_{i}")

hide_all_widgets()
#===========================================


# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from cliphorizontal
cliphorizontalDisplay = Show(cliphorizontal, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'pressure_lim'
pressure_limLUT = GetColorTransferFunction('pressure_lim')
pressure_limLUT.RGBPoints = [0.3, 0.231373, 0.298039, 0.752941, 51.70669090846856, 0.865003, 0.865003, 0.865003, 103.11338181693709, 0.705882, 0.0156863, 0.14902]
pressure_limLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'pressure_lim'
pressure_limPWF = GetOpacityTransferFunction('pressure_lim')
pressure_limPWF.Points = [0.3, 0.0, 0.5, 0.0, 103.11338181693709, 1.0, 0.5, 0.0]
pressure_limPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
cliphorizontalDisplay.Representation = 'Surface'
cliphorizontalDisplay.ColorArrayName = ['CELLS', 'pressure_lim']
cliphorizontalDisplay.LookupTable = pressure_limLUT
cliphorizontalDisplay.SelectTCoordArray = 'None'
cliphorizontalDisplay.SelectNormalArray = 'None'
cliphorizontalDisplay.SelectTangentArray = 'None'
cliphorizontalDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
cliphorizontalDisplay.SelectOrientationVectors = 'None'
cliphorizontalDisplay.ScaleFactor = 7.0
cliphorizontalDisplay.SelectScaleArray = 'None'
cliphorizontalDisplay.GlyphType = 'Arrow'
cliphorizontalDisplay.GlyphTableIndexArray = 'None'
cliphorizontalDisplay.GaussianRadius = 0.35000000000000003
cliphorizontalDisplay.SetScaleArray = ['POINTS', '']
cliphorizontalDisplay.ScaleTransferFunction = 'PiecewiseFunction'
cliphorizontalDisplay.OpacityArray = ['POINTS', '']
cliphorizontalDisplay.OpacityTransferFunction = 'PiecewiseFunction'
cliphorizontalDisplay.DataAxesGrid = 'GridAxesRepresentation'
cliphorizontalDisplay.PolarAxes = 'PolarAxesRepresentation'
cliphorizontalDisplay.ScalarOpacityFunction = pressure_limPWF
cliphorizontalDisplay.ScalarOpacityUnitDistance = 2.2650937418100554
cliphorizontalDisplay.OpacityArrayName = ['CELLS', 'conductivity']

# show data from slicezk40_4
slicezk40_4Display = Show(slicezk40_4, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slicezk40_4Display.Representation = 'Surface'
slicezk40_4Display.ColorArrayName = ['CELLS', 'pressure_lim']
slicezk40_4Display.LookupTable = pressure_limLUT
slicezk40_4Display.SelectTCoordArray = 'None'
slicezk40_4Display.SelectNormalArray = 'None'
slicezk40_4Display.SelectTangentArray = 'None'
slicezk40_4Display.OSPRayScaleFunction = 'PiecewiseFunction'
slicezk40_4Display.SelectOrientationVectors = 'None'
slicezk40_4Display.ScaleFactor = 6.0
slicezk40_4Display.SelectScaleArray = 'None'
slicezk40_4Display.GlyphType = 'Arrow'
slicezk40_4Display.GlyphTableIndexArray = 'None'
slicezk40_4Display.GaussianRadius = 0.3
slicezk40_4Display.SetScaleArray = ['POINTS', '']
slicezk40_4Display.ScaleTransferFunction = 'PiecewiseFunction'
slicezk40_4Display.OpacityArray = ['POINTS', '']
slicezk40_4Display.OpacityTransferFunction = 'PiecewiseFunction'
slicezk40_4Display.DataAxesGrid = 'GridAxesRepresentation'
slicezk40_4Display.PolarAxes = 'PolarAxesRepresentation'

# show data from annotateTimeFilter1
annotateTimeFilter1Display = Show(annotateTimeFilter1, renderView1, 'TextSourceRepresentation')

# setup the color legend parameters for each legend in this view

# get color legend/bar for pressure_limLUT in view renderView1
pressure_limLUTColorBar = GetScalarBar(pressure_limLUT, renderView1)
pressure_limLUTColorBar.WindowLocation = 'Upper Right Corner'
pressure_limLUTColorBar.Title = 'pressure_lim'
pressure_limLUTColorBar.ComponentTitle = ''

# set color bar visibility
pressure_limLUTColorBar.Visibility = 1

# show color legend
cliphorizontalDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
slicezk40_4Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(cliphorizontal)
# ----------------------------------------------------------------



if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')
