# state file generated using paraview version 5.9.1

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
renderView1.ViewSize = [1021, 793]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView1.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView1.CameraViewUp = [0.0, 0.0, 1.0]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 47.762432936357
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [1021, 793]
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView2.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView2.CameraViewUp = [0.0, 0.0, 1.0]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 47.762432936357
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.ViewSize = [1021, 793]
renderView3.AxesGrid = 'GridAxes3DActor'
renderView3.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView3.StereoType = 'Crystal Eyes'
renderView3.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView3.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView3.CameraViewUp = [0.0, 0.0, 1.0]
renderView3.CameraFocalDisk = 1.0
renderView3.CameraParallelScale = 47.762432936357
renderView3.BackEnd = 'OSPRay raycaster'
renderView3.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView4 = CreateView('RenderView')
renderView4.ViewSize = [1021, 793]
renderView4.AxesGrid = 'GridAxes3DActor'
renderView4.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView4.StereoType = 'Crystal Eyes'
renderView4.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView4.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView4.CameraViewUp = [0.0, 0.0, 1.0]
renderView4.CameraFocalDisk = 1.0
renderView4.CameraParallelScale = 47.762432936357
renderView4.BackEnd = 'OSPRay raycaster'
renderView4.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView5 = CreateView('RenderView')
renderView5.ViewSize = [1021, 793]
renderView5.AxesGrid = 'GridAxes3DActor'
renderView5.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView5.StereoType = 'Crystal Eyes'
renderView5.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView5.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView5.CameraViewUp = [0.0, 0.0, 1.0]
renderView5.CameraFocalDisk = 1.0
renderView5.CameraParallelScale = 47.762432936357
renderView5.BackEnd = 'OSPRay raycaster'
renderView5.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView6 = CreateView('RenderView')
renderView6.ViewSize = [1021, 793]
renderView6.AxesGrid = 'GridAxes3DActor'
renderView6.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView6.StereoType = 'Crystal Eyes'
renderView6.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView6.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView6.CameraViewUp = [0.0, 0.0, 1.0]
renderView6.CameraFocalDisk = 1.0
renderView6.CameraParallelScale = 47.762432936357
renderView6.BackEnd = 'OSPRay raycaster'
renderView6.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'max_sample'
max_sample = CreateLayout(name='max_sample')
max_sample.AssignView(0, renderView4)
max_sample.SetSize(1021, 793)

# create new layout object 'mean'
mean = CreateLayout(name='mean')
mean.AssignView(0, renderView5)
mean.SetSize(1021, 793)

# create new layout object 'med_sample'
med_sample = CreateLayout(name='med_sample')
med_sample.AssignView(0, renderView6)
med_sample.SetSize(1021, 793)

# create new layout object 'pressure'
pressure = CreateLayout(name='pressure')
pressure.AssignView(0, renderView2)
pressure.SetSize(1021, 793)

# create new layout object 'pressure_lim'
pressure_lim = CreateLayout(name='pressure_lim')
pressure_lim.AssignView(0, renderView1)
pressure_lim.SetSize(1021, 793)

# create new layout object 'soft_pressure'
soft_pressure = CreateLayout(name='soft_pressure')
soft_pressure.AssignView(0, renderView3)
soft_pressure.SetSize(1021, 793)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView6)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
time_step_vtu = XMLUnstructuredGridReader(registrationName='time_step_..vtu', FileName=['/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_00.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_01.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_02.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_03.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_04.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_05.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_06.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_07.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_08.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_09.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_10.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_11.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_12.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_13.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_14.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_15.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_16.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_17.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_18.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_19.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_20.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_21.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_22.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_23.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_24.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_25.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_26.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_27.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_28.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_29.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_30.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_31.vtu'])
time_step_vtu.CellArrayStatus = ['st_young_modulus', 'st_init_stress_x', 'st_init_stress_y', 'st_init_stress_z', 'st_perm_k0', 'st_perm_eps', 'st_perm_delta', 'st_perm_gamma', 'st_conductivity_a', 'st_conductivity_b', 'st_conductivity_c', 'mean', 'std', 'max_sample', 'med_sample']
time_step_vtu.TimeArray = 'None'

# create a new 'Calculator'
calculator2 = Calculator(registrationName='Calculator2', Input=time_step_vtu)
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

# create a new 'Histogram'
histogram1 = Histogram(registrationName='Histogram1', Input=calculator1)
histogram1.SelectInputArray = ['CELLS', 'st_conductivity_c']
histogram1.BinCount = 20
histogram1.CustomBinRanges = [0.26105615223119116, 361.7054304277176]

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
clip2 = Clip(registrationName='Clip2', Input=time_step_vtu)
clip2.ClipType = 'Plane'
clip2.HyperTreeGridClipper = 'Plane'
clip2.Scalars = ['CELLS', 'st_young_modulus']
clip2.Value = 0.27923714013564455

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [0.0, 0.0, 2.0240356035179]
clip2.ClipType.Normal = [0.0, 0.0, 1.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'pressure_lim'
pressure_limLUT = GetColorTransferFunction('pressure_lim')
pressure_limLUT.RGBPoints = [0.2613621549904437, 0.231373, 0.298039, 0.752941, 342.48278430663277, 0.865003, 0.865003, 0.865003, 684.7042064582752, 0.705882, 0.0156863, 0.14902]
pressure_limLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'pressure_lim'
pressure_limPWF = GetOpacityTransferFunction('pressure_lim')
pressure_limPWF.Points = [0.2613621549904437, 0.0, 0.5, 0.0, 684.7042064582752, 1.0, 0.5, 0.0]
pressure_limPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['CELLS', 'pressure_lim']
clip1Display.LookupTable = pressure_limLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'None'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 7.0
clip1Display.SelectScaleArray = 'pressure_lim'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'pressure_lim'
clip1Display.GaussianRadius = 0.35000000000000003
clip1Display.SetScaleArray = [None, '']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = [None, '']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = pressure_limPWF
clip1Display.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for pressure_limLUT in view renderView1
pressure_limLUTColorBar = GetScalarBar(pressure_limLUT, renderView1)
pressure_limLUTColorBar.Title = 'pressure_lim'
pressure_limLUTColorBar.ComponentTitle = ''

# set color bar visibility
pressure_limLUTColorBar.Visibility = 1

# show color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_1 = Show(clip1, renderView2, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'pressure'
pressureLUT = GetColorTransferFunction('pressure')
pressureLUT.RGBPoints = [-29.39909530445889, 0.231373, 0.298039, 0.752941, 156.47646743635264, 0.865003, 0.865003, 0.865003, 342.3520301771641, 0.705882, 0.0156863, 0.14902]
pressureLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'pressure'
pressurePWF = GetOpacityTransferFunction('pressure')
pressurePWF.Points = [-29.39909530445889, 0.0, 0.5, 0.0, 342.3520301771641, 1.0, 0.5, 0.0]
pressurePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_1.Representation = 'Surface'
clip1Display_1.ColorArrayName = ['CELLS', 'pressure']
clip1Display_1.LookupTable = pressureLUT
clip1Display_1.SelectTCoordArray = 'None'
clip1Display_1.SelectNormalArray = 'None'
clip1Display_1.SelectTangentArray = 'None'
clip1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_1.SelectOrientationVectors = 'None'
clip1Display_1.ScaleFactor = 7.0
clip1Display_1.SelectScaleArray = 'pressure_lim'
clip1Display_1.GlyphType = 'Arrow'
clip1Display_1.GlyphTableIndexArray = 'pressure_lim'
clip1Display_1.GaussianRadius = 0.35000000000000003
clip1Display_1.SetScaleArray = [None, '']
clip1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_1.OpacityArray = [None, '']
clip1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_1.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_1.PolarAxes = 'PolarAxesRepresentation'
clip1Display_1.ScalarOpacityFunction = pressurePWF
clip1Display_1.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_1.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for pressureLUT in view renderView2
pressureLUTColorBar = GetScalarBar(pressureLUT, renderView2)
pressureLUTColorBar.Title = 'pressure'
pressureLUTColorBar.ComponentTitle = ''

# set color bar visibility
pressureLUTColorBar.Visibility = 1

# show color legend
clip1Display_1.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView3'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_2 = Show(clip1, renderView3, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'soft_pressure'
soft_pressureLUT = GetColorTransferFunction('soft_pressure')
soft_pressureLUT.RGBPoints = [60.91059209366698, 0.231373, 0.298039, 0.752941, 372.8073992759711, 0.865003, 0.865003, 0.865003, 684.7042064582752, 0.705882, 0.0156863, 0.14902]
soft_pressureLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'soft_pressure'
soft_pressurePWF = GetOpacityTransferFunction('soft_pressure')
soft_pressurePWF.Points = [60.91059209366698, 0.0, 0.5, 0.0, 684.7042064582752, 1.0, 0.5, 0.0]
soft_pressurePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_2.Representation = 'Surface'
clip1Display_2.ColorArrayName = ['CELLS', 'soft_pressure']
clip1Display_2.LookupTable = soft_pressureLUT
clip1Display_2.SelectTCoordArray = 'None'
clip1Display_2.SelectNormalArray = 'None'
clip1Display_2.SelectTangentArray = 'None'
clip1Display_2.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_2.SelectOrientationVectors = 'None'
clip1Display_2.ScaleFactor = 7.0
clip1Display_2.SelectScaleArray = 'pressure_lim'
clip1Display_2.GlyphType = 'Arrow'
clip1Display_2.GlyphTableIndexArray = 'pressure_lim'
clip1Display_2.GaussianRadius = 0.35000000000000003
clip1Display_2.SetScaleArray = [None, '']
clip1Display_2.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_2.OpacityArray = [None, '']
clip1Display_2.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_2.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_2.PolarAxes = 'PolarAxesRepresentation'
clip1Display_2.ScalarOpacityFunction = soft_pressurePWF
clip1Display_2.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_2.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for soft_pressureLUT in view renderView3
soft_pressureLUTColorBar = GetScalarBar(soft_pressureLUT, renderView3)
soft_pressureLUTColorBar.Title = 'soft_pressure'
soft_pressureLUTColorBar.ComponentTitle = ''

# set color bar visibility
soft_pressureLUTColorBar.Visibility = 1

# show color legend
clip1Display_2.SetScalarBarVisibility(renderView3, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView4'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_3 = Show(clip1, renderView4, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'max_sample'
max_sampleLUT = GetColorTransferFunction('max_sample')
max_sampleLUT.RGBPoints = [9.174413024060243, 0.231373, 0.298039, 0.752941, 138.8855859171455, 0.865003, 0.865003, 0.865003, 268.59675881023077, 0.705882, 0.0156863, 0.14902]
max_sampleLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'max_sample'
max_samplePWF = GetOpacityTransferFunction('max_sample')
max_samplePWF.Points = [9.174413024060243, 0.0, 0.5, 0.0, 268.59675881023077, 1.0, 0.5, 0.0]
max_samplePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_3.Representation = 'Surface'
clip1Display_3.ColorArrayName = ['CELLS', 'max_sample']
clip1Display_3.LookupTable = max_sampleLUT
clip1Display_3.SelectTCoordArray = 'None'
clip1Display_3.SelectNormalArray = 'None'
clip1Display_3.SelectTangentArray = 'None'
clip1Display_3.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_3.SelectOrientationVectors = 'None'
clip1Display_3.ScaleFactor = 7.0
clip1Display_3.SelectScaleArray = 'pressure_lim'
clip1Display_3.GlyphType = 'Arrow'
clip1Display_3.GlyphTableIndexArray = 'pressure_lim'
clip1Display_3.GaussianRadius = 0.35000000000000003
clip1Display_3.SetScaleArray = [None, '']
clip1Display_3.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_3.OpacityArray = [None, '']
clip1Display_3.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_3.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_3.PolarAxes = 'PolarAxesRepresentation'
clip1Display_3.ScalarOpacityFunction = max_samplePWF
clip1Display_3.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_3.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for max_sampleLUT in view renderView4
max_sampleLUTColorBar = GetScalarBar(max_sampleLUT, renderView4)
max_sampleLUTColorBar.Title = 'max_sample'
max_sampleLUTColorBar.ComponentTitle = ''

# set color bar visibility
max_sampleLUTColorBar.Visibility = 1

# show color legend
clip1Display_3.SetScalarBarVisibility(renderView4, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView5'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_4 = Show(clip1, renderView5, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'mean'
meanLUT = GetColorTransferFunction('mean')
meanLUT.RGBPoints = [20.395682058386488, 0.231373, 0.298039, 0.752941, 164.43418504489497, 0.865003, 0.865003, 0.865003, 308.4726880314035, 0.705882, 0.0156863, 0.14902]
meanLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'mean'
meanPWF = GetOpacityTransferFunction('mean')
meanPWF.Points = [20.395682058386488, 0.0, 0.5, 0.0, 308.4726880314035, 1.0, 0.5, 0.0]
meanPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_4.Representation = 'Surface'
clip1Display_4.ColorArrayName = ['CELLS', 'mean']
clip1Display_4.LookupTable = meanLUT
clip1Display_4.SelectTCoordArray = 'None'
clip1Display_4.SelectNormalArray = 'None'
clip1Display_4.SelectTangentArray = 'None'
clip1Display_4.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_4.SelectOrientationVectors = 'None'
clip1Display_4.ScaleFactor = 7.0
clip1Display_4.SelectScaleArray = 'pressure_lim'
clip1Display_4.GlyphType = 'Arrow'
clip1Display_4.GlyphTableIndexArray = 'pressure_lim'
clip1Display_4.GaussianRadius = 0.35000000000000003
clip1Display_4.SetScaleArray = [None, '']
clip1Display_4.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_4.OpacityArray = [None, '']
clip1Display_4.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_4.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_4.PolarAxes = 'PolarAxesRepresentation'
clip1Display_4.ScalarOpacityFunction = meanPWF
clip1Display_4.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_4.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for meanLUT in view renderView5
meanLUTColorBar = GetScalarBar(meanLUT, renderView5)
meanLUTColorBar.Title = 'mean'
meanLUTColorBar.ComponentTitle = ''

# set color bar visibility
meanLUTColorBar.Visibility = 1

# show color legend
clip1Display_4.SetScalarBarVisibility(renderView5, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView6'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_5 = Show(clip1, renderView6, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'med_sample'
med_sampleLUT = GetColorTransferFunction('med_sample')
med_sampleLUT.RGBPoints = [-42.37916112761836, 0.231373, 0.298039, 0.752941, 144.9864345247729, 0.865003, 0.865003, 0.865003, 332.3520301771641, 0.705882, 0.0156863, 0.14902]
med_sampleLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'med_sample'
med_samplePWF = GetOpacityTransferFunction('med_sample')
med_samplePWF.Points = [-42.37916112761836, 0.0, 0.5, 0.0, 332.3520301771641, 1.0, 0.5, 0.0]
med_samplePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_5.Representation = 'Surface'
clip1Display_5.ColorArrayName = ['CELLS', 'med_sample']
clip1Display_5.LookupTable = med_sampleLUT
clip1Display_5.SelectTCoordArray = 'None'
clip1Display_5.SelectNormalArray = 'None'
clip1Display_5.SelectTangentArray = 'None'
clip1Display_5.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_5.SelectOrientationVectors = 'None'
clip1Display_5.ScaleFactor = 7.0
clip1Display_5.SelectScaleArray = 'pressure_lim'
clip1Display_5.GlyphType = 'Arrow'
clip1Display_5.GlyphTableIndexArray = 'pressure_lim'
clip1Display_5.GaussianRadius = 0.35000000000000003
clip1Display_5.SetScaleArray = [None, '']
clip1Display_5.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_5.OpacityArray = [None, '']
clip1Display_5.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_5.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_5.PolarAxes = 'PolarAxesRepresentation'
clip1Display_5.ScalarOpacityFunction = med_samplePWF
clip1Display_5.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_5.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for med_sampleLUT in view renderView6
med_sampleLUTColorBar = GetScalarBar(med_sampleLUT, renderView6)
med_sampleLUTColorBar.Title = 'med_sample'
med_sampleLUTColorBar.ComponentTitle = ''

# set color bar visibility
med_sampleLUTColorBar.Visibility = 1

# show color legend
clip1Display_5.SetScalarBarVisibility(renderView6, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(clip1)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')