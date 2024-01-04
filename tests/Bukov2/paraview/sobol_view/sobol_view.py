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

# Create a new 'Bar Chart View'
barChartView1 = CreateView('XYBarChartView')
barChartView1.ViewSize = [1085, 793]
barChartView1.LegendPosition = [743, 751]
barChartView1.LeftAxisUseCustomRange = 1
barChartView1.LeftAxisRangeMaximum = 0.01
barChartView1.BottomAxisRangeMaximum = 0.00034
barChartView1.RightAxisRangeMaximum = 6.66
barChartView1.TopAxisRangeMaximum = 6.66

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [1085, 793]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.0, 0.0, 2.301543157765501]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [175.36920297572397, -194.69584762782287, 31.66122243358308]
renderView1.CameraFocalPoint = [-5.276112217847983e-17, -7.489396580419807e-17, 2.3015431577655]
renderView1.CameraViewUp = [-0.0682677108956331, 0.08836353891371196, 0.9937461469810664]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 68.24328501194424
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView1.AxesGrid.Visibility = 1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1085, 793)

# create new layout object 'Layout #2'
layout2 = CreateLayout(name='Layout #2')
layout2.AssignView(0, barChartView1)
layout2.SetSize(1085, 793)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
time_step_ = XMLUnstructuredGridReader(registrationName='time_step_*', FileName=['/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_00.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_01.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_02.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_03.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_04.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_05.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_06.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_07.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_08.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_09.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_10.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_11.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_12.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_13.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_14.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_15.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_16.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_17.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_18.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_19.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_20.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_21.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_22.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_23.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_24.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_25.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_26.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_27.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_28.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_29.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_30.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_01_02/sensitivity/time_step_31.vtu'])
time_step_.CellArrayStatus = ['st_young_modulus', 'st_init_stress_x', 'st_init_stress_y', 'st_init_stress_z', 'st_perm_k0', 'st_perm_eps', 'st_perm_delta', 'st_perm_gamma', 'st_conductivity_a', 'st_conductivity_b', 'st_conductivity_c', 'mean', 'std', 'max_sample', 'med_sample']
time_step_.TimeArray = 'None'

# create a new 'Calculator'
calculator2 = Calculator(registrationName='Calculator2', Input=time_step_)
calculator2.AttributeType = 'Cell Data'
calculator2.ResultArrayName = 'pressure'
calculator2.Function = 'med_sample + 10'

# create a new 'Calculator'
calculator3 = Calculator(registrationName='Calculator3', Input=calculator2)
calculator3.AttributeType = 'Cell Data'
calculator3.ResultArrayName = 'soft_pressure'
calculator3.Function = '0.13 + pressure +sqrt((0.13-pressure)^2 + 0.1)'

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=calculator3)
calculator1.AttributeType = 'Cell Data'
calculator1.ResultArrayName = 'pressure_lim'
calculator1.Function = 'soft_pressure'

# create a new 'Histogram'
histogram1 = Histogram(registrationName='Histogram1', Input=calculator1)
histogram1.SelectInputArray = ['CELLS', 'st_conductivity_c']
histogram1.BinCount = 20
histogram1.Normalize = 1
histogram1.CustomBinRanges = [0.26105615223119116, 361.7054304277176]

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=calculator1)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['CELLS', 'pressure_lim']
clip1.Value = 180.98324328997438
clip1.Invert = 0

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.0, 5.0, 0.0]
clip1.ClipType.Normal = [0.0, 1.0, 0.0]

# create a new 'Clip'
clipX4 = Clip(registrationName='Clip X - (-4)', Input=calculator1)
clipX4.ClipType = 'Plane'
clipX4.HyperTreeGridClipper = 'Plane'
clipX4.Scalars = ['CELLS', 'st_young_modulus']
clipX4.Value = 0.3525947484323372

# init the 'Plane' selected for 'ClipType'
clipX4.ClipType.Origin = [-4.0, 0.0, 0.0]

# create a new 'Clip'
clipX8 = Clip(registrationName='Clip X - (-8)', Input=calculator1)
clipX8.ClipType = 'Plane'
clipX8.HyperTreeGridClipper = 'Plane'
clipX8.Scalars = ['CELLS', 'st_young_modulus']
clipX8.Value = 0.3525947484323372

# init the 'Plane' selected for 'ClipType'
clipX8.ClipType.Origin = [-8.0, 0.0, 0.0]

# create a new 'Clip'
clipXmain = Clip(registrationName='Clip X - main', Input=calculator1)
clipXmain.ClipType = 'Plane'
clipXmain.HyperTreeGridClipper = 'Plane'
clipXmain.Scalars = ['CELLS', 'st_young_modulus']

# create a new 'Clip'
clip2 = Clip(registrationName='Clip2', Input=time_step_)
clip2.ClipType = 'Plane'
clip2.HyperTreeGridClipper = 'Plane'
clip2.Scalars = ['CELLS', 'st_young_modulus']
clip2.Value = 0.27923714013564455

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [0.0, 0.0, 2.0240356035179]
clip2.ClipType.Normal = [0.0, 0.0, 1.0]

# ----------------------------------------------------------------
# setup the visualization in view 'barChartView1'
# ----------------------------------------------------------------

# show data from histogram1
histogram1Display = Show(histogram1, barChartView1, 'XYBarChartRepresentation')

# trace defaults for the display properties.
histogram1Display.AttributeType = 'Row Data'
histogram1Display.UseIndexForXAxis = 0
histogram1Display.XArrayName = 'bin_extents'
histogram1Display.SeriesVisibility = ['bin_values']
histogram1Display.SeriesLabel = ['bin_extents', 'bin_extents', 'bin_values', 'bin_values']
histogram1Display.SeriesColor = ['bin_extents', '0', '0', '0', 'bin_values', '0.8899977111467154', '0.10000762951094835', '0.1100022888532845']
histogram1Display.SeriesPlotCorner = ['bin_extents', '0', 'bin_values', '0']
histogram1Display.SeriesLabelPrefix = ''

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from clipX4
clipX4Display = Show(clipX4, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'st_init_stress_y'
st_init_stress_yLUT = GetColorTransferFunction('st_init_stress_y')
st_init_stress_yLUT.AutomaticRescaleRangeMode = 'Never'
st_init_stress_yLUT.RGBPoints = [0.01, 0.278431372549, 0.278431372549, 0.858823529412, 0.01931968317016925, 0.0, 0.0, 0.360784313725, 0.03715352290971724, 0.0, 1.0, 1.0, 0.07211074791828997, 0.0, 0.501960784314, 0.0, 0.13867558288718881, 1.0, 1.0, 0.0, 0.2679168324819033, 1.0, 0.380392156863, 0.0, 0.5176068319505673, 0.419607843137, 0.0, 0.0, 1.0, 0.878431372549, 0.301960784314, 0.301960784314]
st_init_stress_yLUT.UseLogScale = 1
st_init_stress_yLUT.ColorSpace = 'RGB'
st_init_stress_yLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'st_init_stress_y'
st_init_stress_yPWF = GetOpacityTransferFunction('st_init_stress_y')
st_init_stress_yPWF.Points = [0.01, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
st_init_stress_yPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clipX4Display.Representation = 'Surface'
clipX4Display.ColorArrayName = ['CELLS', 'st_init_stress_y']
clipX4Display.LookupTable = st_init_stress_yLUT
clipX4Display.SelectTCoordArray = 'None'
clipX4Display.SelectNormalArray = 'None'
clipX4Display.SelectTangentArray = 'None'
clipX4Display.OSPRayScaleFunction = 'PiecewiseFunction'
clipX4Display.SelectOrientationVectors = 'None'
clipX4Display.ScaleFactor = 6.0
clipX4Display.SelectScaleArray = 'st_young_modulus'
clipX4Display.GlyphType = 'Arrow'
clipX4Display.GlyphTableIndexArray = 'st_young_modulus'
clipX4Display.GaussianRadius = 0.3
clipX4Display.SetScaleArray = [None, '']
clipX4Display.ScaleTransferFunction = 'PiecewiseFunction'
clipX4Display.OpacityArray = [None, '']
clipX4Display.OpacityTransferFunction = 'PiecewiseFunction'
clipX4Display.DataAxesGrid = 'GridAxesRepresentation'
clipX4Display.PolarAxes = 'PolarAxesRepresentation'
clipX4Display.ScalarOpacityFunction = st_init_stress_yPWF
clipX4Display.ScalarOpacityUnitDistance = 3.9497304179107497
clipX4Display.OpacityArrayName = ['CELLS', 'st_young_modulus']

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'st_perm_eps'
st_perm_epsLUT = GetColorTransferFunction('st_perm_eps')
st_perm_epsLUT.RGBPoints = [1.369789000471124e-11, 0.278431372549, 0.278431372549, 0.858823529412, 2.727785435280252e-10, 0.0, 0.0, 0.360784313725, 5.319633682475571e-09, 0.0, 1.0, 1.0, 1.0817409477693672e-07, 0.0, 0.501960784314, 0.0, 2.1095741281702933e-06, 1.0, 1.0, 0.0, 4.200986852345709e-05, 1.0, 0.380392156863, 0.0, 0.0008365807248920195, 0.419607843137, 0.0, 0.0, 0.01665959294469281, 0.878431372549, 0.301960784314, 0.301960784314]
st_perm_epsLUT.UseLogScale = 1
st_perm_epsLUT.ColorSpace = 'RGB'
st_perm_epsLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'st_perm_eps'
st_perm_epsPWF = GetOpacityTransferFunction('st_perm_eps')
st_perm_epsPWF.Points = [1.3697890004711232e-11, 0.0, 0.5, 0.0, 0.016659592944692835, 1.0, 0.5, 0.0]
st_perm_epsPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['CELLS', 'st_perm_eps']
clip1Display.LookupTable = st_perm_epsLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'None'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 6.000000000000003
clip1Display.SelectScaleArray = 'pressure_lim'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'pressure_lim'
clip1Display.GaussianRadius = 0.3000000000000001
clip1Display.SetScaleArray = [None, '']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = [None, '']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = st_perm_epsPWF
clip1Display.ScalarOpacityUnitDistance = 3.599114582628065
clip1Display.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for st_init_stress_yLUT in view renderView1
st_init_stress_yLUTColorBar = GetScalarBar(st_init_stress_yLUT, renderView1)
st_init_stress_yLUTColorBar.Title = 'st_init_stress_y'
st_init_stress_yLUTColorBar.ComponentTitle = ''

# set color bar visibility
st_init_stress_yLUTColorBar.Visibility = 1

# get color legend/bar for st_perm_epsLUT in view renderView1
st_perm_epsLUTColorBar = GetScalarBar(st_perm_epsLUT, renderView1)
st_perm_epsLUTColorBar.WindowLocation = 'Upper Right Corner'
st_perm_epsLUTColorBar.Title = 'st_perm_eps'
st_perm_epsLUTColorBar.ComponentTitle = ''

# set color bar visibility
st_perm_epsLUTColorBar.Visibility = 1

# show color legend
clipX4Display.SetScalarBarVisibility(renderView1, True)

# show color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(time_step_)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')