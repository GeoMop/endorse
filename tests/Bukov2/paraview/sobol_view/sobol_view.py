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
renderView1.ViewSize = [640, 480]
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
renderView10 = CreateView('RenderView')
renderView10.ViewSize = [640, 480]
renderView10.AxesGrid = 'GridAxes3DActor'
renderView10.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView10.StereoType = 'Crystal Eyes'
renderView10.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView10.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView10.CameraViewUp = [0.0, 0.0, 1.0]
renderView10.CameraFocalDisk = 1.0
renderView10.CameraParallelScale = 47.762432936357
renderView10.BackEnd = 'OSPRay raycaster'
renderView10.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView11 = CreateView('RenderView')
renderView11.ViewSize = [640, 480]
renderView11.AxesGrid = 'GridAxes3DActor'
renderView11.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView11.StereoType = 'Crystal Eyes'
renderView11.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView11.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView11.CameraViewUp = [0.0, 0.0, 1.0]
renderView11.CameraFocalDisk = 1.0
renderView11.CameraParallelScale = 47.762432936357
renderView11.BackEnd = 'OSPRay raycaster'
renderView11.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView12 = CreateView('RenderView')
renderView12.ViewSize = [640, 480]
renderView12.AxesGrid = 'GridAxes3DActor'
renderView12.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView12.StereoType = 'Crystal Eyes'
renderView12.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView12.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView12.CameraViewUp = [0.0, 0.0, 1.0]
renderView12.CameraFocalDisk = 1.0
renderView12.CameraParallelScale = 47.762432936357
renderView12.BackEnd = 'OSPRay raycaster'
renderView12.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView13 = CreateView('RenderView')
renderView13.ViewSize = [640, 480]
renderView13.AxesGrid = 'GridAxes3DActor'
renderView13.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView13.StereoType = 'Crystal Eyes'
renderView13.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView13.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView13.CameraViewUp = [0.0, 0.0, 1.0]
renderView13.CameraFocalDisk = 1.0
renderView13.CameraParallelScale = 47.762432936357
renderView13.BackEnd = 'OSPRay raycaster'
renderView13.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView14 = CreateView('RenderView')
renderView14.ViewSize = [640, 480]
renderView14.AxesGrid = 'GridAxes3DActor'
renderView14.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView14.StereoType = 'Crystal Eyes'
renderView14.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView14.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView14.CameraViewUp = [0.0, 0.0, 1.0]
renderView14.CameraFocalDisk = 1.0
renderView14.CameraParallelScale = 47.762432936357
renderView14.BackEnd = 'OSPRay raycaster'
renderView14.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView15 = CreateView('RenderView')
renderView15.ViewSize = [640, 480]
renderView15.AxesGrid = 'GridAxes3DActor'
renderView15.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView15.StereoType = 'Crystal Eyes'
renderView15.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView15.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView15.CameraViewUp = [0.0, 0.0, 1.0]
renderView15.CameraFocalDisk = 1.0
renderView15.CameraParallelScale = 47.762432936357
renderView15.BackEnd = 'OSPRay raycaster'
renderView15.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView16 = CreateView('RenderView')
renderView16.ViewSize = [1021, 793]
renderView16.AxesGrid = 'GridAxes3DActor'
renderView16.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView16.StereoType = 'Crystal Eyes'
renderView16.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView16.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView16.CameraViewUp = [0.0, 0.0, 1.0]
renderView16.CameraFocalDisk = 1.0
renderView16.CameraParallelScale = 47.762432936357
renderView16.BackEnd = 'OSPRay raycaster'
renderView16.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView17 = CreateView('RenderView')
renderView17.ViewSize = [640, 480]
renderView17.AxesGrid = 'GridAxes3DActor'
renderView17.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView17.StereoType = 'Crystal Eyes'
renderView17.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView17.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView17.CameraViewUp = [0.0, 0.0, 1.0]
renderView17.CameraFocalDisk = 1.0
renderView17.CameraParallelScale = 47.762432936357
renderView17.BackEnd = 'OSPRay raycaster'
renderView17.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView18 = CreateView('RenderView')
renderView18.ViewSize = [1021, 793]
renderView18.AxesGrid = 'GridAxes3DActor'
renderView18.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView18.StereoType = 'Crystal Eyes'
renderView18.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView18.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView18.CameraViewUp = [0.0, 0.0, 1.0]
renderView18.CameraFocalDisk = 1.0
renderView18.CameraParallelScale = 47.762432936357
renderView18.BackEnd = 'OSPRay raycaster'
renderView18.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [640, 480]
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
renderView3.ViewSize = [640, 480]
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
renderView4.ViewSize = [640, 480]
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
renderView5.ViewSize = [640, 480]
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
renderView6.ViewSize = [640, 480]
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

# Create a new 'Render View'
renderView7 = CreateView('RenderView')
renderView7.ViewSize = [1021, 793]
renderView7.AxesGrid = 'GridAxes3DActor'
renderView7.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView7.StereoType = 'Crystal Eyes'
renderView7.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView7.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView7.CameraViewUp = [0.0, 0.0, 1.0]
renderView7.CameraFocalDisk = 1.0
renderView7.CameraParallelScale = 47.762432936357
renderView7.BackEnd = 'OSPRay raycaster'
renderView7.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView8 = CreateView('RenderView')
renderView8.ViewSize = [1021, 793]
renderView8.AxesGrid = 'GridAxes3DActor'
renderView8.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView8.StereoType = 'Crystal Eyes'
renderView8.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView8.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView8.CameraViewUp = [0.0, 0.0, 1.0]
renderView8.CameraFocalDisk = 1.0
renderView8.CameraParallelScale = 47.762432936357
renderView8.BackEnd = 'OSPRay raycaster'
renderView8.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView9 = CreateView('RenderView')
renderView9.ViewSize = [1021, 793]
renderView9.AxesGrid = 'GridAxes3DActor'
renderView9.CenterOfRotation = [0.0, 17.500000000000004, 0.0]
renderView9.StereoType = 'Crystal Eyes'
renderView9.CameraPosition = [0.0, -167.0398699985074, 0.0]
renderView9.CameraFocalPoint = [0.0, 17.500000000000004, 0.0]
renderView9.CameraViewUp = [0.0, 0.0, 1.0]
renderView9.CameraFocalDisk = 1.0
renderView9.CameraParallelScale = 47.762432936357
renderView9.BackEnd = 'OSPRay raycaster'
renderView9.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'cond_a'
cond_a = CreateLayout(name='cond_a')
cond_a.AssignView(0, renderView16)
cond_a.SetSize(1021, 793)

# create new layout object 'cond_b'
cond_b = CreateLayout(name='cond_b')
cond_b.AssignView(0, renderView17)
cond_b.SetSize(640, 480)

# create new layout object 'cond_c'
cond_c = CreateLayout(name='cond_c')
cond_c.AssignView(0, renderView18)
cond_c.SetSize(1021, 793)

# create new layout object 'init_pressure'
init_pressure = CreateLayout(name='init_pressure')
init_pressure.AssignView(0, renderView7)
init_pressure.SetSize(1021, 793)

# create new layout object 'init_stress_x'
init_stress_x = CreateLayout(name='init_stress_x')
init_stress_x.AssignView(0, renderView8)
init_stress_x.SetSize(1021, 793)

# create new layout object 'init_stress_y'
init_stress_y = CreateLayout(name='init_stress_y')
init_stress_y.AssignView(0, renderView9)
init_stress_y.SetSize(1021, 793)

# create new layout object 'init_stress_z'
init_stress_z = CreateLayout(name='init_stress_z')
init_stress_z.AssignView(0, renderView10)
init_stress_z.SetSize(640, 480)

# create new layout object 'max_sample'
max_sample = CreateLayout(name='max_sample')
max_sample.AssignView(0, renderView4)
max_sample.SetSize(640, 480)

# create new layout object 'mean'
mean = CreateLayout(name='mean')
mean.AssignView(0, renderView5)
mean.SetSize(640, 480)

# create new layout object 'med_sample'
med_sample = CreateLayout(name='med_sample')
med_sample.AssignView(0, renderView6)
med_sample.SetSize(640, 480)

# create new layout object 'perm_delta'
perm_delta = CreateLayout(name='perm_delta')
perm_delta.AssignView(0, renderView14)
perm_delta.SetSize(640, 480)

# create new layout object 'perm_eps'
perm_eps = CreateLayout(name='perm_eps')
perm_eps.AssignView(0, renderView13)
perm_eps.SetSize(640, 480)

# create new layout object 'perm_gamma'
perm_gamma = CreateLayout(name='perm_gamma')
perm_gamma.AssignView(0, renderView15)
perm_gamma.SetSize(640, 480)

# create new layout object 'perm_k0'
perm_k0 = CreateLayout(name='perm_k0')
perm_k0.AssignView(0, renderView12)
perm_k0.SetSize(640, 480)

# create new layout object 'pressure'
pressure = CreateLayout(name='pressure')
pressure.AssignView(0, renderView2)
pressure.SetSize(640, 480)

# create new layout object 'pressure_lim'
pressure_lim = CreateLayout(name='pressure_lim')
pressure_lim.AssignView(0, renderView1)
pressure_lim.SetSize(640, 480)

# create new layout object 'soft_pressure'
soft_pressure = CreateLayout(name='soft_pressure')
soft_pressure.AssignView(0, renderView3)
soft_pressure.SetSize(640, 480)

# create new layout object 'young_modulus'
young_modulus = CreateLayout(name='young_modulus')
young_modulus.AssignView(0, renderView11)
young_modulus.SetSize(640, 480)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView18)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
time_step_vtu = XMLUnstructuredGridReader(registrationName='time_step_..vtu', FileName=['/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_00.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_01.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_02.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_03.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_04.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_05.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_06.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_07.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_08.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_09.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_10.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_11.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_12.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_13.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_14.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_15.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_16.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_17.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_18.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_19.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_20.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_21.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_22.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_23.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_24.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_25.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_26.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_27.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_28.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_29.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_30.vtu', '/home/paulie/Workspace/endorse-experiment/tests/sandbox/20240102175607_3d_model_D02/sensitivity/time_step_31.vtu'])
time_step_vtu.CellArrayStatus = ['st_young_modulus', 'st_init_stress_x', 'st_init_stress_y', 'st_init_stress_z', 'st_perm_k0', 'st_perm_eps', 'st_perm_delta', 'st_perm_gamma', 'st_conductivity_a', 'st_conductivity_b', 'st_conductivity_c', 'mean', 'std', 'max_sample', 'med_sample']
time_step_vtu.TimeArray = 'None'

# create a new 'Clip'
clip2 = Clip(registrationName='Clip2', Input=time_step_vtu)
clip2.ClipType = 'Plane'
clip2.HyperTreeGridClipper = 'Plane'
clip2.Scalars = ['CELLS', 'st_young_modulus']
clip2.Value = 0.27923714013564455

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [0.0, 0.0, 2.0240356035179]
clip2.ClipType.Normal = [0.0, 0.0, 1.0]

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
clipX4 = Clip(registrationName='Clip X - (-4)', Input=calculator1)
clipX4.ClipType = 'Plane'
clipX4.HyperTreeGridClipper = 'Plane'
clipX4.Scalars = ['CELLS', 'st_young_modulus']
clipX4.Value = 0.3525947484323372

# init the 'Plane' selected for 'ClipType'
clipX4.ClipType.Origin = [-4.0, 0.0, 0.0]

# create a new 'Clip'
clipXmain = Clip(registrationName='Clip X - main', Input=calculator1)
clipXmain.ClipType = 'Plane'
clipXmain.HyperTreeGridClipper = 'Plane'
clipXmain.Scalars = ['CELLS', 'st_young_modulus']

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
# setup the visualization in view 'renderView10'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_1 = Show(clip1, renderView10, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'init_stress_z'
init_stress_zLUT = GetColorTransferFunction('init_stress_z')
init_stress_zLUT.AutomaticRescaleRangeMode = 'Never'
init_stress_zLUT.RGBPoints = [0.01, 0.231373, 0.298039, 0.752941, 0.255, 0.865003, 0.865003, 0.865003, 0.5, 0.705882, 0.0156863, 0.14902]
init_stress_zLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'init_stress_z'
init_stress_zPWF = GetOpacityTransferFunction('init_stress_z')
init_stress_zPWF.Points = [0.01, 0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]
init_stress_zPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_1.Representation = 'Surface'
clip1Display_1.ColorArrayName = ['CELLS', 'init_stress_z']
clip1Display_1.LookupTable = init_stress_zLUT
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
clip1Display_1.ScalarOpacityFunction = init_stress_zPWF
clip1Display_1.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_1.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for init_stress_zLUT in view renderView10
init_stress_zLUTColorBar = GetScalarBar(init_stress_zLUT, renderView10)
init_stress_zLUTColorBar.Title = 'init_stress_z'
init_stress_zLUTColorBar.ComponentTitle = ''

# set color bar visibility
init_stress_zLUTColorBar.Visibility = 1

# show color legend
clip1Display_1.SetScalarBarVisibility(renderView10, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView11'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_2 = Show(clip1, renderView11, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'young_modulus'
young_modulusLUT = GetColorTransferFunction('young_modulus')
young_modulusLUT.AutomaticRescaleRangeMode = 'Never'
young_modulusLUT.RGBPoints = [0.01, 0.231373, 0.298039, 0.752941, 0.255, 0.865003, 0.865003, 0.865003, 0.5, 0.705882, 0.0156863, 0.14902]
young_modulusLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'young_modulus'
young_modulusPWF = GetOpacityTransferFunction('young_modulus')
young_modulusPWF.Points = [0.01, 0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]
young_modulusPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_2.Representation = 'Surface'
clip1Display_2.ColorArrayName = ['CELLS', 'young_modulus']
clip1Display_2.LookupTable = young_modulusLUT
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
clip1Display_2.ScalarOpacityFunction = young_modulusPWF
clip1Display_2.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_2.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for young_modulusLUT in view renderView11
young_modulusLUTColorBar = GetScalarBar(young_modulusLUT, renderView11)
young_modulusLUTColorBar.Title = 'young_modulus'
young_modulusLUTColorBar.ComponentTitle = ''

# set color bar visibility
young_modulusLUTColorBar.Visibility = 1

# show color legend
clip1Display_2.SetScalarBarVisibility(renderView11, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView12'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_3 = Show(clip1, renderView12, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'perm_k0'
perm_k0LUT = GetColorTransferFunction('perm_k0')
perm_k0LUT.AutomaticRescaleRangeMode = 'Never'
perm_k0LUT.RGBPoints = [0.01, 0.231373, 0.298039, 0.752941, 0.505, 0.865003, 0.865003, 0.865003, 1.0, 0.705882, 0.0156863, 0.14902]
perm_k0LUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'perm_k0'
perm_k0PWF = GetOpacityTransferFunction('perm_k0')
perm_k0PWF.Points = [0.01, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
perm_k0PWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_3.Representation = 'Surface'
clip1Display_3.ColorArrayName = ['CELLS', 'perm_k0']
clip1Display_3.LookupTable = perm_k0LUT
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
clip1Display_3.ScalarOpacityFunction = perm_k0PWF
clip1Display_3.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_3.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for perm_k0LUT in view renderView12
perm_k0LUTColorBar = GetScalarBar(perm_k0LUT, renderView12)
perm_k0LUTColorBar.Title = 'perm_k0'
perm_k0LUTColorBar.ComponentTitle = ''

# set color bar visibility
perm_k0LUTColorBar.Visibility = 1

# show color legend
clip1Display_3.SetScalarBarVisibility(renderView12, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView13'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_4 = Show(clip1, renderView13, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'perm_eps'
perm_epsLUT = GetColorTransferFunction('perm_eps')
perm_epsLUT.RGBPoints = [4.803834321444003e-10, 0.231373, 0.298039, 0.752941, 0.010563208433313707, 0.865003, 0.865003, 0.865003, 0.021126416386243983, 0.705882, 0.0156863, 0.14902]
perm_epsLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'perm_eps'
perm_epsPWF = GetOpacityTransferFunction('perm_eps')
perm_epsPWF.Points = [4.803834321444003e-10, 0.0, 0.5, 0.0, 0.021126416386243983, 1.0, 0.5, 0.0]
perm_epsPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_4.Representation = 'Surface'
clip1Display_4.ColorArrayName = ['CELLS', 'perm_eps']
clip1Display_4.LookupTable = perm_epsLUT
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
clip1Display_4.ScalarOpacityFunction = perm_epsPWF
clip1Display_4.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_4.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for perm_epsLUT in view renderView13
perm_epsLUTColorBar = GetScalarBar(perm_epsLUT, renderView13)
perm_epsLUTColorBar.Title = 'perm_eps'
perm_epsLUTColorBar.ComponentTitle = ''

# set color bar visibility
perm_epsLUTColorBar.Visibility = 1

# show color legend
clip1Display_4.SetScalarBarVisibility(renderView13, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView14'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_5 = Show(clip1, renderView14, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'perm_delta'
perm_deltaLUT = GetColorTransferFunction('perm_delta')
perm_deltaLUT.RGBPoints = [6.12295578735106e-11, 0.231373, 0.298039, 0.752941, 0.00033168965339600844, 0.865003, 0.865003, 0.865003, 0.000663379245562459, 0.705882, 0.0156863, 0.14902]
perm_deltaLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'perm_delta'
perm_deltaPWF = GetOpacityTransferFunction('perm_delta')
perm_deltaPWF.Points = [6.12295578735106e-11, 0.0, 0.5, 0.0, 0.000663379245562459, 1.0, 0.5, 0.0]
perm_deltaPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_5.Representation = 'Surface'
clip1Display_5.ColorArrayName = ['CELLS', 'perm_delta']
clip1Display_5.LookupTable = perm_deltaLUT
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
clip1Display_5.ScalarOpacityFunction = perm_deltaPWF
clip1Display_5.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_5.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for perm_deltaLUT in view renderView14
perm_deltaLUTColorBar = GetScalarBar(perm_deltaLUT, renderView14)
perm_deltaLUTColorBar.Title = 'perm_delta'
perm_deltaLUTColorBar.ComponentTitle = ''

# set color bar visibility
perm_deltaLUTColorBar.Visibility = 1

# show color legend
clip1Display_5.SetScalarBarVisibility(renderView14, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView15'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_6 = Show(clip1, renderView15, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'perm_gamma'
perm_gammaLUT = GetColorTransferFunction('perm_gamma')
perm_gammaLUT.RGBPoints = [5.110017571453316e-08, 0.231373, 0.298039, 0.752941, 0.006749749253540457, 0.865003, 0.865003, 0.865003, 0.0134994474069052, 0.705882, 0.0156863, 0.14902]
perm_gammaLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'perm_gamma'
perm_gammaPWF = GetOpacityTransferFunction('perm_gamma')
perm_gammaPWF.Points = [5.110017571453316e-08, 0.0, 0.5, 0.0, 0.0134994474069052, 1.0, 0.5, 0.0]
perm_gammaPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_6.Representation = 'Surface'
clip1Display_6.ColorArrayName = ['CELLS', 'perm_gamma']
clip1Display_6.LookupTable = perm_gammaLUT
clip1Display_6.SelectTCoordArray = 'None'
clip1Display_6.SelectNormalArray = 'None'
clip1Display_6.SelectTangentArray = 'None'
clip1Display_6.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_6.SelectOrientationVectors = 'None'
clip1Display_6.ScaleFactor = 7.0
clip1Display_6.SelectScaleArray = 'pressure_lim'
clip1Display_6.GlyphType = 'Arrow'
clip1Display_6.GlyphTableIndexArray = 'pressure_lim'
clip1Display_6.GaussianRadius = 0.35000000000000003
clip1Display_6.SetScaleArray = [None, '']
clip1Display_6.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_6.OpacityArray = [None, '']
clip1Display_6.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_6.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_6.PolarAxes = 'PolarAxesRepresentation'
clip1Display_6.ScalarOpacityFunction = perm_gammaPWF
clip1Display_6.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_6.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for perm_gammaLUT in view renderView15
perm_gammaLUTColorBar = GetScalarBar(perm_gammaLUT, renderView15)
perm_gammaLUTColorBar.Title = 'perm_gamma'
perm_gammaLUTColorBar.ComponentTitle = ''

# set color bar visibility
perm_gammaLUTColorBar.Visibility = 1

# show color legend
clip1Display_6.SetScalarBarVisibility(renderView15, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView16'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_7 = Show(clip1, renderView16, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'conductivity_a'
conductivity_aLUT = GetColorTransferFunction('conductivity_a')
conductivity_aLUT.RGBPoints = [1.3356988671733807e-09, 0.231373, 0.298039, 0.752941, 0.00027905661606798766, 0.865003, 0.865003, 0.865003, 0.0005581118964371082, 0.705882, 0.0156863, 0.14902]
conductivity_aLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'conductivity_a'
conductivity_aPWF = GetOpacityTransferFunction('conductivity_a')
conductivity_aPWF.Points = [1.3356988671733807e-09, 0.0, 0.5, 0.0, 0.0005581118964371082, 1.0, 0.5, 0.0]
conductivity_aPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_7.Representation = 'Surface'
clip1Display_7.ColorArrayName = ['CELLS', 'conductivity_a']
clip1Display_7.LookupTable = conductivity_aLUT
clip1Display_7.SelectTCoordArray = 'None'
clip1Display_7.SelectNormalArray = 'None'
clip1Display_7.SelectTangentArray = 'None'
clip1Display_7.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_7.SelectOrientationVectors = 'None'
clip1Display_7.ScaleFactor = 7.0
clip1Display_7.SelectScaleArray = 'pressure_lim'
clip1Display_7.GlyphType = 'Arrow'
clip1Display_7.GlyphTableIndexArray = 'pressure_lim'
clip1Display_7.GaussianRadius = 0.35000000000000003
clip1Display_7.SetScaleArray = [None, '']
clip1Display_7.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_7.OpacityArray = [None, '']
clip1Display_7.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_7.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_7.PolarAxes = 'PolarAxesRepresentation'
clip1Display_7.ScalarOpacityFunction = conductivity_aPWF
clip1Display_7.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_7.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for conductivity_aLUT in view renderView16
conductivity_aLUTColorBar = GetScalarBar(conductivity_aLUT, renderView16)
conductivity_aLUTColorBar.Title = 'conductivity_a'
conductivity_aLUTColorBar.ComponentTitle = ''

# set color bar visibility
conductivity_aLUTColorBar.Visibility = 1

# show color legend
clip1Display_7.SetScalarBarVisibility(renderView16, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView17'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_8 = Show(clip1, renderView17, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'conductivity_b'
conductivity_bLUT = GetColorTransferFunction('conductivity_b')
conductivity_bLUT.RGBPoints = [4.012014566116475e-08, 0.231373, 0.298039, 0.752941, 0.00039085364421652207, 0.865003, 0.865003, 0.865003, 0.000781667168287383, 0.705882, 0.0156863, 0.14902]
conductivity_bLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'conductivity_b'
conductivity_bPWF = GetOpacityTransferFunction('conductivity_b')
conductivity_bPWF.Points = [4.012014566116475e-08, 0.0, 0.5, 0.0, 0.000781667168287383, 1.0, 0.5, 0.0]
conductivity_bPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_8.Representation = 'Surface'
clip1Display_8.ColorArrayName = ['CELLS', 'conductivity_b']
clip1Display_8.LookupTable = conductivity_bLUT
clip1Display_8.SelectTCoordArray = 'None'
clip1Display_8.SelectNormalArray = 'None'
clip1Display_8.SelectTangentArray = 'None'
clip1Display_8.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_8.SelectOrientationVectors = 'None'
clip1Display_8.ScaleFactor = 7.0
clip1Display_8.SelectScaleArray = 'pressure_lim'
clip1Display_8.GlyphType = 'Arrow'
clip1Display_8.GlyphTableIndexArray = 'pressure_lim'
clip1Display_8.GaussianRadius = 0.35000000000000003
clip1Display_8.SetScaleArray = [None, '']
clip1Display_8.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_8.OpacityArray = [None, '']
clip1Display_8.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_8.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_8.PolarAxes = 'PolarAxesRepresentation'
clip1Display_8.ScalarOpacityFunction = conductivity_bPWF
clip1Display_8.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_8.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for conductivity_bLUT in view renderView17
conductivity_bLUTColorBar = GetScalarBar(conductivity_bLUT, renderView17)
conductivity_bLUTColorBar.Title = 'conductivity_b'
conductivity_bLUTColorBar.ComponentTitle = ''

# set color bar visibility
conductivity_bLUTColorBar.Visibility = 1

# show color legend
clip1Display_8.SetScalarBarVisibility(renderView17, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView18'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_9 = Show(clip1, renderView18, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'conductivity_c'
conductivity_cLUT = GetColorTransferFunction('conductivity_c')
conductivity_cLUT.RGBPoints = [5.234779614783583e-09, 0.231373, 0.298039, 0.752941, 0.0002707702867308218, 0.865003, 0.865003, 0.865003, 0.0005415353386820288, 0.705882, 0.0156863, 0.14902]
conductivity_cLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'conductivity_c'
conductivity_cPWF = GetOpacityTransferFunction('conductivity_c')
conductivity_cPWF.Points = [5.234779614783583e-09, 0.0, 0.5, 0.0, 0.0005415353386820288, 1.0, 0.5, 0.0]
conductivity_cPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_9.Representation = 'Surface'
clip1Display_9.ColorArrayName = ['CELLS', 'conductivity_c']
clip1Display_9.LookupTable = conductivity_cLUT
clip1Display_9.SelectTCoordArray = 'None'
clip1Display_9.SelectNormalArray = 'None'
clip1Display_9.SelectTangentArray = 'None'
clip1Display_9.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_9.SelectOrientationVectors = 'None'
clip1Display_9.ScaleFactor = 7.0
clip1Display_9.SelectScaleArray = 'pressure_lim'
clip1Display_9.GlyphType = 'Arrow'
clip1Display_9.GlyphTableIndexArray = 'pressure_lim'
clip1Display_9.GaussianRadius = 0.35000000000000003
clip1Display_9.SetScaleArray = [None, '']
clip1Display_9.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_9.OpacityArray = [None, '']
clip1Display_9.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_9.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_9.PolarAxes = 'PolarAxesRepresentation'
clip1Display_9.ScalarOpacityFunction = conductivity_cPWF
clip1Display_9.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_9.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for conductivity_cLUT in view renderView18
conductivity_cLUTColorBar = GetScalarBar(conductivity_cLUT, renderView18)
conductivity_cLUTColorBar.Title = 'conductivity_c'
conductivity_cLUTColorBar.ComponentTitle = ''

# set color bar visibility
conductivity_cLUTColorBar.Visibility = 1

# show color legend
clip1Display_9.SetScalarBarVisibility(renderView18, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_10 = Show(clip1, renderView2, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'pressure'
pressureLUT = GetColorTransferFunction('pressure')
pressureLUT.RGBPoints = [-29.39909530445889, 0.231373, 0.298039, 0.752941, 156.47646743635264, 0.865003, 0.865003, 0.865003, 342.3520301771641, 0.705882, 0.0156863, 0.14902]
pressureLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'pressure'
pressurePWF = GetOpacityTransferFunction('pressure')
pressurePWF.Points = [-29.39909530445889, 0.0, 0.5, 0.0, 342.3520301771641, 1.0, 0.5, 0.0]
pressurePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_10.Representation = 'Surface'
clip1Display_10.ColorArrayName = ['CELLS', 'pressure']
clip1Display_10.LookupTable = pressureLUT
clip1Display_10.SelectTCoordArray = 'None'
clip1Display_10.SelectNormalArray = 'None'
clip1Display_10.SelectTangentArray = 'None'
clip1Display_10.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_10.SelectOrientationVectors = 'None'
clip1Display_10.ScaleFactor = 7.0
clip1Display_10.SelectScaleArray = 'pressure_lim'
clip1Display_10.GlyphType = 'Arrow'
clip1Display_10.GlyphTableIndexArray = 'pressure_lim'
clip1Display_10.GaussianRadius = 0.35000000000000003
clip1Display_10.SetScaleArray = [None, '']
clip1Display_10.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_10.OpacityArray = [None, '']
clip1Display_10.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_10.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_10.PolarAxes = 'PolarAxesRepresentation'
clip1Display_10.ScalarOpacityFunction = pressurePWF
clip1Display_10.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_10.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for pressureLUT in view renderView2
pressureLUTColorBar = GetScalarBar(pressureLUT, renderView2)
pressureLUTColorBar.Title = 'pressure'
pressureLUTColorBar.ComponentTitle = ''

# set color bar visibility
pressureLUTColorBar.Visibility = 1

# show color legend
clip1Display_10.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView3'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_11 = Show(clip1, renderView3, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'soft_pressure'
soft_pressureLUT = GetColorTransferFunction('soft_pressure')
soft_pressureLUT.RGBPoints = [60.91059209366698, 0.231373, 0.298039, 0.752941, 372.8073992759711, 0.865003, 0.865003, 0.865003, 684.7042064582752, 0.705882, 0.0156863, 0.14902]
soft_pressureLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'soft_pressure'
soft_pressurePWF = GetOpacityTransferFunction('soft_pressure')
soft_pressurePWF.Points = [60.91059209366698, 0.0, 0.5, 0.0, 684.7042064582752, 1.0, 0.5, 0.0]
soft_pressurePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_11.Representation = 'Surface'
clip1Display_11.ColorArrayName = ['CELLS', 'soft_pressure']
clip1Display_11.LookupTable = soft_pressureLUT
clip1Display_11.SelectTCoordArray = 'None'
clip1Display_11.SelectNormalArray = 'None'
clip1Display_11.SelectTangentArray = 'None'
clip1Display_11.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_11.SelectOrientationVectors = 'None'
clip1Display_11.ScaleFactor = 7.0
clip1Display_11.SelectScaleArray = 'pressure_lim'
clip1Display_11.GlyphType = 'Arrow'
clip1Display_11.GlyphTableIndexArray = 'pressure_lim'
clip1Display_11.GaussianRadius = 0.35000000000000003
clip1Display_11.SetScaleArray = [None, '']
clip1Display_11.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_11.OpacityArray = [None, '']
clip1Display_11.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_11.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_11.PolarAxes = 'PolarAxesRepresentation'
clip1Display_11.ScalarOpacityFunction = soft_pressurePWF
clip1Display_11.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_11.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for soft_pressureLUT in view renderView3
soft_pressureLUTColorBar = GetScalarBar(soft_pressureLUT, renderView3)
soft_pressureLUTColorBar.Title = 'soft_pressure'
soft_pressureLUTColorBar.ComponentTitle = ''

# set color bar visibility
soft_pressureLUTColorBar.Visibility = 1

# show color legend
clip1Display_11.SetScalarBarVisibility(renderView3, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView4'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_12 = Show(clip1, renderView4, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'max_sample'
max_sampleLUT = GetColorTransferFunction('max_sample')
max_sampleLUT.RGBPoints = [9.174413024060243, 0.231373, 0.298039, 0.752941, 138.8855859171455, 0.865003, 0.865003, 0.865003, 268.59675881023077, 0.705882, 0.0156863, 0.14902]
max_sampleLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'max_sample'
max_samplePWF = GetOpacityTransferFunction('max_sample')
max_samplePWF.Points = [9.174413024060243, 0.0, 0.5, 0.0, 268.59675881023077, 1.0, 0.5, 0.0]
max_samplePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_12.Representation = 'Surface'
clip1Display_12.ColorArrayName = ['CELLS', 'max_sample']
clip1Display_12.LookupTable = max_sampleLUT
clip1Display_12.SelectTCoordArray = 'None'
clip1Display_12.SelectNormalArray = 'None'
clip1Display_12.SelectTangentArray = 'None'
clip1Display_12.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_12.SelectOrientationVectors = 'None'
clip1Display_12.ScaleFactor = 7.0
clip1Display_12.SelectScaleArray = 'pressure_lim'
clip1Display_12.GlyphType = 'Arrow'
clip1Display_12.GlyphTableIndexArray = 'pressure_lim'
clip1Display_12.GaussianRadius = 0.35000000000000003
clip1Display_12.SetScaleArray = [None, '']
clip1Display_12.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_12.OpacityArray = [None, '']
clip1Display_12.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_12.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_12.PolarAxes = 'PolarAxesRepresentation'
clip1Display_12.ScalarOpacityFunction = max_samplePWF
clip1Display_12.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_12.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for max_sampleLUT in view renderView4
max_sampleLUTColorBar = GetScalarBar(max_sampleLUT, renderView4)
max_sampleLUTColorBar.Title = 'max_sample'
max_sampleLUTColorBar.ComponentTitle = ''

# set color bar visibility
max_sampleLUTColorBar.Visibility = 1

# show color legend
clip1Display_12.SetScalarBarVisibility(renderView4, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView5'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_13 = Show(clip1, renderView5, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'mean'
meanLUT = GetColorTransferFunction('mean')
meanLUT.RGBPoints = [20.395682058386488, 0.231373, 0.298039, 0.752941, 164.43418504489497, 0.865003, 0.865003, 0.865003, 308.4726880314035, 0.705882, 0.0156863, 0.14902]
meanLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'mean'
meanPWF = GetOpacityTransferFunction('mean')
meanPWF.Points = [20.395682058386488, 0.0, 0.5, 0.0, 308.4726880314035, 1.0, 0.5, 0.0]
meanPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_13.Representation = 'Surface'
clip1Display_13.ColorArrayName = ['CELLS', 'mean']
clip1Display_13.LookupTable = meanLUT
clip1Display_13.SelectTCoordArray = 'None'
clip1Display_13.SelectNormalArray = 'None'
clip1Display_13.SelectTangentArray = 'None'
clip1Display_13.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_13.SelectOrientationVectors = 'None'
clip1Display_13.ScaleFactor = 7.0
clip1Display_13.SelectScaleArray = 'pressure_lim'
clip1Display_13.GlyphType = 'Arrow'
clip1Display_13.GlyphTableIndexArray = 'pressure_lim'
clip1Display_13.GaussianRadius = 0.35000000000000003
clip1Display_13.SetScaleArray = [None, '']
clip1Display_13.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_13.OpacityArray = [None, '']
clip1Display_13.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_13.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_13.PolarAxes = 'PolarAxesRepresentation'
clip1Display_13.ScalarOpacityFunction = meanPWF
clip1Display_13.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_13.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for meanLUT in view renderView5
meanLUTColorBar = GetScalarBar(meanLUT, renderView5)
meanLUTColorBar.Title = 'mean'
meanLUTColorBar.ComponentTitle = ''

# set color bar visibility
meanLUTColorBar.Visibility = 1

# show color legend
clip1Display_13.SetScalarBarVisibility(renderView5, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView6'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_14 = Show(clip1, renderView6, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'med_sample'
med_sampleLUT = GetColorTransferFunction('med_sample')
med_sampleLUT.RGBPoints = [-42.37916112761836, 0.231373, 0.298039, 0.752941, 144.9864345247729, 0.865003, 0.865003, 0.865003, 332.3520301771641, 0.705882, 0.0156863, 0.14902]
med_sampleLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'med_sample'
med_samplePWF = GetOpacityTransferFunction('med_sample')
med_samplePWF.Points = [-42.37916112761836, 0.0, 0.5, 0.0, 332.3520301771641, 1.0, 0.5, 0.0]
med_samplePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_14.Representation = 'Surface'
clip1Display_14.ColorArrayName = ['CELLS', 'med_sample']
clip1Display_14.LookupTable = med_sampleLUT
clip1Display_14.SelectTCoordArray = 'None'
clip1Display_14.SelectNormalArray = 'None'
clip1Display_14.SelectTangentArray = 'None'
clip1Display_14.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_14.SelectOrientationVectors = 'None'
clip1Display_14.ScaleFactor = 7.0
clip1Display_14.SelectScaleArray = 'pressure_lim'
clip1Display_14.GlyphType = 'Arrow'
clip1Display_14.GlyphTableIndexArray = 'pressure_lim'
clip1Display_14.GaussianRadius = 0.35000000000000003
clip1Display_14.SetScaleArray = [None, '']
clip1Display_14.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_14.OpacityArray = [None, '']
clip1Display_14.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_14.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_14.PolarAxes = 'PolarAxesRepresentation'
clip1Display_14.ScalarOpacityFunction = med_samplePWF
clip1Display_14.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_14.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for med_sampleLUT in view renderView6
med_sampleLUTColorBar = GetScalarBar(med_sampleLUT, renderView6)
med_sampleLUTColorBar.Title = 'med_sample'
med_sampleLUTColorBar.ComponentTitle = ''

# set color bar visibility
med_sampleLUTColorBar.Visibility = 1

# show color legend
clip1Display_14.SetScalarBarVisibility(renderView6, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView7'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_15 = Show(clip1, renderView7, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'init_pressure'
init_pressureLUT = GetColorTransferFunction('init_pressure')
init_pressureLUT.AutomaticRescaleRangeMode = 'Never'
init_pressureLUT.RGBPoints = [0.01, 0.231373, 0.298039, 0.752941, 0.505, 0.865003, 0.865003, 0.865003, 1.0, 0.705882, 0.0156863, 0.14902]
init_pressureLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'init_pressure'
init_pressurePWF = GetOpacityTransferFunction('init_pressure')
init_pressurePWF.Points = [0.01, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
init_pressurePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_15.Representation = 'Surface'
clip1Display_15.ColorArrayName = ['CELLS', 'init_pressure']
clip1Display_15.LookupTable = init_pressureLUT
clip1Display_15.SelectTCoordArray = 'None'
clip1Display_15.SelectNormalArray = 'None'
clip1Display_15.SelectTangentArray = 'None'
clip1Display_15.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_15.SelectOrientationVectors = 'None'
clip1Display_15.ScaleFactor = 7.0
clip1Display_15.SelectScaleArray = 'pressure_lim'
clip1Display_15.GlyphType = 'Arrow'
clip1Display_15.GlyphTableIndexArray = 'pressure_lim'
clip1Display_15.GaussianRadius = 0.35000000000000003
clip1Display_15.SetScaleArray = [None, '']
clip1Display_15.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_15.OpacityArray = [None, '']
clip1Display_15.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_15.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_15.PolarAxes = 'PolarAxesRepresentation'
clip1Display_15.ScalarOpacityFunction = init_pressurePWF
clip1Display_15.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_15.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for init_pressureLUT in view renderView7
init_pressureLUTColorBar = GetScalarBar(init_pressureLUT, renderView7)
init_pressureLUTColorBar.Title = 'init_pressure'
init_pressureLUTColorBar.ComponentTitle = ''

# set color bar visibility
init_pressureLUTColorBar.Visibility = 1

# show color legend
clip1Display_15.SetScalarBarVisibility(renderView7, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView8'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_16 = Show(clip1, renderView8, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'init_stress_x'
init_stress_xLUT = GetColorTransferFunction('init_stress_x')
init_stress_xLUT.AutomaticRescaleRangeMode = 'Never'
init_stress_xLUT.RGBPoints = [0.01, 0.231373, 0.298039, 0.752941, 0.255, 0.865003, 0.865003, 0.865003, 0.5, 0.705882, 0.0156863, 0.14902]
init_stress_xLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'init_stress_x'
init_stress_xPWF = GetOpacityTransferFunction('init_stress_x')
init_stress_xPWF.Points = [0.01, 0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]
init_stress_xPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_16.Representation = 'Surface'
clip1Display_16.ColorArrayName = ['CELLS', 'init_stress_x']
clip1Display_16.LookupTable = init_stress_xLUT
clip1Display_16.SelectTCoordArray = 'None'
clip1Display_16.SelectNormalArray = 'None'
clip1Display_16.SelectTangentArray = 'None'
clip1Display_16.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_16.SelectOrientationVectors = 'None'
clip1Display_16.ScaleFactor = 7.0
clip1Display_16.SelectScaleArray = 'pressure_lim'
clip1Display_16.GlyphType = 'Arrow'
clip1Display_16.GlyphTableIndexArray = 'pressure_lim'
clip1Display_16.GaussianRadius = 0.35000000000000003
clip1Display_16.SetScaleArray = [None, '']
clip1Display_16.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_16.OpacityArray = [None, '']
clip1Display_16.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_16.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_16.PolarAxes = 'PolarAxesRepresentation'
clip1Display_16.ScalarOpacityFunction = init_stress_xPWF
clip1Display_16.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_16.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for init_stress_xLUT in view renderView8
init_stress_xLUTColorBar = GetScalarBar(init_stress_xLUT, renderView8)
init_stress_xLUTColorBar.Title = 'init_stress_x'
init_stress_xLUTColorBar.ComponentTitle = ''

# set color bar visibility
init_stress_xLUTColorBar.Visibility = 1

# show color legend
clip1Display_16.SetScalarBarVisibility(renderView8, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView9'
# ----------------------------------------------------------------

# show data from clip1
clip1Display_17 = Show(clip1, renderView9, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'init_stress_y'
init_stress_yLUT = GetColorTransferFunction('init_stress_y')
init_stress_yLUT.AutomaticRescaleRangeMode = 'Never'
init_stress_yLUT.RGBPoints = [0.01, 0.231373, 0.298039, 0.752941, 0.255, 0.865003, 0.865003, 0.865003, 0.5, 0.705882, 0.0156863, 0.14902]
init_stress_yLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'init_stress_y'
init_stress_yPWF = GetOpacityTransferFunction('init_stress_y')
init_stress_yPWF.Points = [0.01, 0.0, 0.5, 0.0, 0.5, 1.0, 0.5, 0.0]
init_stress_yPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display_17.Representation = 'Surface'
clip1Display_17.ColorArrayName = ['CELLS', 'init_stress_y']
clip1Display_17.LookupTable = init_stress_yLUT
clip1Display_17.SelectTCoordArray = 'None'
clip1Display_17.SelectNormalArray = 'None'
clip1Display_17.SelectTangentArray = 'None'
clip1Display_17.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display_17.SelectOrientationVectors = 'None'
clip1Display_17.ScaleFactor = 7.0
clip1Display_17.SelectScaleArray = 'pressure_lim'
clip1Display_17.GlyphType = 'Arrow'
clip1Display_17.GlyphTableIndexArray = 'pressure_lim'
clip1Display_17.GaussianRadius = 0.35000000000000003
clip1Display_17.SetScaleArray = [None, '']
clip1Display_17.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display_17.OpacityArray = [None, '']
clip1Display_17.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display_17.DataAxesGrid = 'GridAxesRepresentation'
clip1Display_17.PolarAxes = 'PolarAxesRepresentation'
clip1Display_17.ScalarOpacityFunction = init_stress_yPWF
clip1Display_17.ScalarOpacityUnitDistance = 4.423720263251087
clip1Display_17.OpacityArrayName = ['CELLS', 'pressure_lim']

# setup the color legend parameters for each legend in this view

# get color legend/bar for init_stress_yLUT in view renderView9
init_stress_yLUTColorBar = GetScalarBar(init_stress_yLUT, renderView9)
init_stress_yLUTColorBar.Title = 'init_stress_y'
init_stress_yLUTColorBar.ComponentTitle = ''

# set color bar visibility
init_stress_yLUTColorBar.Visibility = 1

# show color legend
clip1Display_17.SetScalarBarVisibility(renderView9, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(None)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')