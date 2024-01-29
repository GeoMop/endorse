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

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')
lineChartView1.ViewSize = [858, 793]
lineChartView1.LeftAxisRangeMaximum = 6.66
lineChartView1.BottomAxisRangeMaximum = 6.66
lineChartView1.RightAxisRangeMaximum = 6.66
lineChartView1.TopAxisRangeMaximum = 6.66

# Create a new 'Line Chart View'
lineChartView2 = CreateView('XYChartView')
lineChartView2.ViewSize = [858, 793]
lineChartView2.LeftAxisLogScale = 1
lineChartView2.LeftAxisUseCustomRange = 1
lineChartView2.LeftAxisRangeMinimum = 2.438953857577173e-22
lineChartView2.LeftAxisRangeMaximum = 4100.118568841651
lineChartView2.BottomAxisUseCustomRange = 1
lineChartView2.BottomAxisRangeMinimum = -82330892.8
lineChartView2.BottomAxisRangeMaximum = 32330892.799999997
lineChartView2.RightAxisUseCustomRange = 1
lineChartView2.RightAxisRangeMinimum = -4.30647492096
lineChartView2.RightAxisRangeMaximum = 10.966474920960003
lineChartView2.TopAxisUseCustomRange = 1
lineChartView2.TopAxisRangeMinimum = -4.30647492096
lineChartView2.TopAxisRangeMaximum = 10.966474920960003

# get the material library
materialLibrary1 = GetMaterialLibrary()

# Create a new 'Render View'
renderView1 = CreateView('RenderView')
renderView1.ViewSize = [858, 793]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-8.86181484600183, -136.2739532126084, 25.097247446549616]
renderView1.CameraFocalPoint = [8.088548306595259, 73.29205854088339, -5.7667555069593295]
renderView1.CameraViewUp = [-0.006245402322603242, 0.14619567233457334, 0.9892359780863569]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 55.00000000000001
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [858, 793]
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [13.703474167040586, 80.74686855401903, 2.1399616266696735]
renderView2.CameraViewUp = [0.5512104911730297, -0.11544517908724697, 0.8263409738396742]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 55.00000000000001
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.ViewSize = [858, 793]
renderView3.AxesGrid = 'GridAxes3DActor'
renderView3.StereoType = 'Crystal Eyes'
renderView3.CameraPosition = [-0.9216517048011558, 212.46655181470683, 3.8638956620182956]
renderView3.CameraViewUp = [0.7400314527844074, -0.009020074278042894, 0.6725117747294973]
renderView3.CameraFocalDisk = 1.0
renderView3.CameraParallelScale = 55.00000000000001
renderView3.BackEnd = 'OSPRay raycaster'
renderView3.OSPRayMaterialLibrary = materialLibrary1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(858, 793)

# create new layout object 'Layout #2'
layout2 = CreateLayout(name='Layout #2')
layout2.AssignView(0, renderView2)
layout2.SetSize(858, 793)

# create new layout object 'Layout #3'
layout3 = CreateLayout(name='Layout #3')
layout3.AssignView(0, renderView3)
layout3.SetSize(858, 793)

# create new layout object 'Layout #4'
layout4 = CreateLayout(name='Layout #4')
layout4.AssignView(0, lineChartView1)
layout4.SetSize(858, 793)

# create new layout object 'Layout #5'
layout5 = CreateLayout(name='Layout #5')
layout5.AssignView(0, lineChartView2)
layout5.SetSize(858, 793)

# ----------------------------------------------------------------
# restore active view
SetActiveView(lineChartView2)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML Unstructured Grid Reader'
time_step_ = XMLUnstructuredGridReader(registrationName='time_step_*', FileName=['/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_00.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_01.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_02.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_03.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_04.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_4.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_05.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_5.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_06.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_6.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_07.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_7.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_08.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_8.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_09.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_9.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_10.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_11.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_12.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_13.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_14.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_15.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_16.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_17.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_18.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_19.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_20.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_21.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_22.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_23.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_24.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_25.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_26.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_27.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_28.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_29.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_30.vtu', '/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_12_29_full/sensitivity/time_step_31.vtu'])
time_step_.CellArrayStatus = ['st_young_modulus', 'st_init_stress_x', 'st_init_stress_y', 'st_init_stress_z', 'st_perm_k0', 'st_perm_eps', 'st_perm_delta', 'st_perm_gamma', 'st_conductivity_a', 'st_conductivity_b', 'st_conductivity_c', 'mean', 'std', 'max_sample', 'med_sample']
time_step_.TimeArray = 'None'

# create a new 'PVD Reader'
mechanicspvd = PVDReader(registrationName='mechanics.pvd', FileName='/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_samples/solver_00_sample_000/output_D02_hm/mechanics.pvd')
mechanicspvd.CellArrays = ['initial_stress', 'region_id', 'stress', 'von_mises_stress', 'mean_stress']
mechanicspvd.PointArrays = ['displacement']

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=mechanicspvd)
calculator1.AttributeType = 'Cell Data'
calculator1.Function = 'von_mises_stress - 55e6'

# create a new 'Clip'
clip2 = Clip(registrationName='Clip2', Input=time_step_)
clip2.ClipType = 'Plane'
clip2.HyperTreeGridClipper = 'Plane'
clip2.Scalars = ['CELLS', 'st_young_modulus']
clip2.Value = 0.00021859736072362315
clip2.Invert = 0

# init the 'Plane' selected for 'ClipType'
clip2.ClipType.Origin = [0.0, 5.0509722577040455, 0.0]
clip2.ClipType.Normal = [0.0, 1.0, 0.0]

# create a new 'Clip'
clip4 = Clip(registrationName='Clip4', Input=mechanicspvd)
clip4.ClipType = 'Plane'
clip4.HyperTreeGridClipper = 'Plane'
clip4.Scalars = ['CELLS', 'mean_stress']
clip4.Value = -25503079.311404906

# init the 'Plane' selected for 'ClipType'
clip4.ClipType.Origin = [-9.293357644821533, 0.0, 0.0]

# create a new 'PVD Reader'
flowpvd = PVDReader(registrationName='flow.pvd', FileName='/home/jb/workspace/endorse-experiment/tests/Bukov2/3d_model/PE_samples/solver_00_sample_000/output_D02_hm/flow.pvd')
flowpvd.CellArrays = ['region_id', 'pressure_p0', 'conductivity']

# create a new 'Clip'
clip5 = Clip(registrationName='Clip5', Input=flowpvd)
clip5.ClipType = 'Plane'
clip5.HyperTreeGridClipper = 'Plane'
clip5.Scalars = ['CELLS', 'conductivity']
clip5.Value = 2.9714072391115497e-06

# create a new 'Append Attributes'
appendAttributes1 = AppendAttributes(registrationName='AppendAttributes1', Input=[flowpvd, mechanicspvd])

# create a new 'Calculator'
calculator2 = Calculator(registrationName='Calculator2', Input=appendAttributes1)
calculator2.AttributeType = 'Cell Data'
calculator2.ResultArrayName = 'cond_mean_stress'
calculator2.Function = '1000*9.81/0.001 * (7.5e-22 + 1e-16*exp( -8*  mean_stress/23e6))'

# create a new 'Calculator'
calculator3 = Calculator(registrationName='Calculator3', Input=calculator2)
calculator3.AttributeType = 'Cell Data'
calculator3.ResultArrayName = 'vm_exp'
calculator3.Function = '(von_mises_stress-55e6)/55e6'

# create a new 'Calculator'
calculator4 = Calculator(registrationName='Calculator4', Input=calculator3)
calculator4.AttributeType = 'Cell Data'
calculator4.Function = 'cond_mean_stress*exp(6*(abs(vm_exp)+vm_exp))'

# create a new 'Plot Data'
plotData1 = PlotData(registrationName='PlotData1', Input=calculator4)

# create a new 'Clip'
clip1 = Clip(registrationName='Clip1', Input=time_step_)
clip1.ClipType = 'Plane'
clip1.HyperTreeGridClipper = 'Plane'
clip1.Scalars = ['CELLS', 'st_young_modulus']
clip1.Value = 5.3932312180436505e-11

# init the 'Plane' selected for 'ClipType'
clip1.ClipType.Origin = [0.4932216493226924, 0.0, 1.3601467665551263]
clip1.ClipType.Normal = [0.0, 0.0, 1.0]

# create a new 'Clip'
clip3 = Clip(registrationName='Clip3', Input=flowpvd)
clip3.ClipType = 'Plane'
clip3.HyperTreeGridClipper = 'Plane'
clip3.Scalars = ['CELLS', 'conductivity']
clip3.Value = 1.897884585916652e-07

# init the 'Plane' selected for 'ClipType'
clip3.ClipType.Origin = [-8.324648559126693, 0.0, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'lineChartView2'
# ----------------------------------------------------------------

# show data from plotData1
plotData1Display = Show(plotData1, lineChartView2, 'XYChartRepresentation')

# trace defaults for the display properties.
plotData1Display.AttributeType = 'Cell Data'
plotData1Display.UseIndexForXAxis = 0
plotData1Display.XArrayName = 'mean_stress'
plotData1Display.SeriesVisibility = ['cond_mean_stress', 'conductivity']
plotData1Display.SeriesLabel = ['Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude', 'conductivity', 'conductivity', 'initial_stress_0', 'initial_stress_0', 'initial_stress_1', 'initial_stress_1', 'initial_stress_2', 'initial_stress_2', 'initial_stress_3', 'initial_stress_3', 'initial_stress_4', 'initial_stress_4', 'initial_stress_5', 'initial_stress_5', 'initial_stress_6', 'initial_stress_6', 'initial_stress_7', 'initial_stress_7', 'initial_stress_8', 'initial_stress_8', 'initial_stress_Magnitude', 'initial_stress_Magnitude', 'mean_stress', 'mean_stress', 'pressure_p0', 'pressure_p0', 'region_id', 'region_id', 'region_id_input_1', 'region_id_input_1', 'stress_0', 'stress_0', 'stress_1', 'stress_1', 'stress_2', 'stress_2', 'stress_3', 'stress_3', 'stress_4', 'stress_4', 'stress_5', 'stress_5', 'stress_6', 'stress_6', 'stress_7', 'stress_7', 'stress_8', 'stress_8', 'stress_Magnitude', 'stress_Magnitude', 'von_mises_stress', 'von_mises_stress', 'cond_mean_stress', 'cond_mean_stress', 'Result', '1cond', 'vm_exp', 'vm_exp']
plotData1Display.SeriesColor = ['Points_X', '0', '0', '0', 'Points_Y', '0.8941176470588236', '0.10196078431372549', '0.10980392156862745', 'Points_Z', '0.21568627450980393', '0.49411764705882355', '0.7215686274509804', 'Points_Magnitude', '0.30196078431372547', '0.6862745098039216', '0.2901960784313726', 'conductivity', '0.596078431372549', '0.3058823529411765', '0.6392156862745098', 'initial_stress_0', '1', '0.4980392156862745', '0', 'initial_stress_1', '0.6509803921568628', '0.33725490196078434', '0.1568627450980392', 'initial_stress_2', '0', '0', '0', 'initial_stress_3', '0.8941176470588236', '0.10196078431372549', '0.10980392156862745', 'initial_stress_4', '0.21568627450980393', '0.49411764705882355', '0.7215686274509804', 'initial_stress_5', '0.30196078431372547', '0.6862745098039216', '0.2901960784313726', 'initial_stress_6', '0.596078431372549', '0.3058823529411765', '0.6392156862745098', 'initial_stress_7', '1', '0.4980392156862745', '0', 'initial_stress_8', '0.6509803921568628', '0.33725490196078434', '0.1568627450980392', 'initial_stress_Magnitude', '0', '0', '0', 'mean_stress', '0.8941176470588236', '0.10196078431372549', '0.10980392156862745', 'pressure_p0', '0.21568627450980393', '0.49411764705882355', '0.7215686274509804', 'region_id', '0.30196078431372547', '0.6862745098039216', '0.2901960784313726', 'region_id_input_1', '0.596078431372549', '0.3058823529411765', '0.6392156862745098', 'stress_0', '1', '0.4980392156862745', '0', 'stress_1', '0.6509803921568628', '0.33725490196078434', '0.1568627450980392', 'stress_2', '0', '0', '0', 'stress_3', '0.8941176470588236', '0.10196078431372549', '0.10980392156862745', 'stress_4', '0.21568627450980393', '0.49411764705882355', '0.7215686274509804', 'stress_5', '0.30196078431372547', '0.6862745098039216', '0.2901960784313726', 'stress_6', '0.596078431372549', '0.3058823529411765', '0.6392156862745098', 'stress_7', '1', '0.4980392156862745', '0', 'stress_8', '0.6509803921568628', '0.33725490196078434', '0.1568627450980392', 'stress_Magnitude', '0', '0', '0', 'von_mises_stress', '0.8941176470588236', '0.10196078431372549', '0.10980392156862745', 'cond_mean_stress', '0.21568627450980393', '0.49411764705882355', '0.7215686274509804', 'Result', '0.30196078431372547', '0.6862745098039216', '0.2901960784313726', 'vm_exp', '0.596078431372549', '0.3058823529411765', '0.6392156862745098']
plotData1Display.SeriesPlotCorner = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Result', '0', 'cond_mean_stress', '0', 'conductivity', '0', 'initial_stress_0', '0', 'initial_stress_1', '0', 'initial_stress_2', '0', 'initial_stress_3', '0', 'initial_stress_4', '0', 'initial_stress_5', '0', 'initial_stress_6', '0', 'initial_stress_7', '0', 'initial_stress_8', '0', 'initial_stress_Magnitude', '0', 'mean_stress', '0', 'pressure_p0', '0', 'region_id', '0', 'region_id_input_1', '0', 'stress_0', '0', 'stress_1', '0', 'stress_2', '0', 'stress_3', '0', 'stress_4', '0', 'stress_5', '0', 'stress_6', '0', 'stress_7', '0', 'stress_8', '0', 'stress_Magnitude', '0', 'vm_exp', '0', 'von_mises_stress', '0']
plotData1Display.SeriesLabelPrefix = ''
plotData1Display.SeriesLineStyle = ['Points_Magnitude', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Result', '1', 'cond_mean_stress', '1', 'conductivity', '1', 'initial_stress_0', '1', 'initial_stress_1', '1', 'initial_stress_2', '1', 'initial_stress_3', '1', 'initial_stress_4', '1', 'initial_stress_5', '1', 'initial_stress_6', '1', 'initial_stress_7', '1', 'initial_stress_8', '1', 'initial_stress_Magnitude', '1', 'mean_stress', '1', 'pressure_p0', '1', 'region_id', '1', 'region_id_input_1', '1', 'stress_0', '1', 'stress_1', '1', 'stress_2', '1', 'stress_3', '1', 'stress_4', '1', 'stress_5', '1', 'stress_6', '1', 'stress_7', '1', 'stress_8', '1', 'stress_Magnitude', '1', 'vm_exp', '1', 'von_mises_stress', '1']
plotData1Display.SeriesLineThickness = ['Points_Magnitude', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Result', '1', 'cond_mean_stress', '2', 'conductivity', '2', 'initial_stress_0', '2', 'initial_stress_1', '2', 'initial_stress_2', '2', 'initial_stress_3', '2', 'initial_stress_4', '2', 'initial_stress_5', '2', 'initial_stress_6', '2', 'initial_stress_7', '2', 'initial_stress_8', '2', 'initial_stress_Magnitude', '2', 'mean_stress', '2', 'pressure_p0', '2', 'region_id', '2', 'region_id_input_1', '2', 'stress_0', '2', 'stress_1', '2', 'stress_2', '2', 'stress_3', '2', 'stress_4', '2', 'stress_5', '2', 'stress_6', '2', 'stress_7', '2', 'stress_8', '2', 'stress_Magnitude', '2', 'vm_exp', '2', 'von_mises_stress', '2']
plotData1Display.SeriesMarkerStyle = ['Points_Magnitude', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Result', '0', 'cond_mean_stress', '0', 'conductivity', '0', 'initial_stress_0', '0', 'initial_stress_1', '0', 'initial_stress_2', '0', 'initial_stress_3', '0', 'initial_stress_4', '0', 'initial_stress_5', '0', 'initial_stress_6', '0', 'initial_stress_7', '0', 'initial_stress_8', '0', 'initial_stress_Magnitude', '0', 'mean_stress', '0', 'pressure_p0', '0', 'region_id', '0', 'region_id_input_1', '0', 'stress_0', '0', 'stress_1', '0', 'stress_2', '0', 'stress_3', '0', 'stress_4', '0', 'stress_5', '0', 'stress_6', '0', 'stress_7', '0', 'stress_8', '0', 'stress_Magnitude', '0', 'vm_exp', '0', 'von_mises_stress', '0']
plotData1Display.SeriesMarkerSize = ['Points_Magnitude', '4', 'Points_X', '4', 'Points_Y', '4', 'Points_Z', '4', 'Result', '4', 'cond_mean_stress', '4', 'conductivity', '4', 'initial_stress_0', '4', 'initial_stress_1', '4', 'initial_stress_2', '4', 'initial_stress_3', '4', 'initial_stress_4', '4', 'initial_stress_5', '4', 'initial_stress_6', '4', 'initial_stress_7', '4', 'initial_stress_8', '4', 'initial_stress_Magnitude', '4', 'mean_stress', '4', 'pressure_p0', '4', 'region_id', '4', 'region_id_input_1', '4', 'stress_0', '4', 'stress_1', '4', 'stress_2', '4', 'stress_3', '4', 'stress_4', '4', 'stress_5', '4', 'stress_6', '4', 'stress_7', '4', 'stress_8', '4', 'stress_Magnitude', '4', 'vm_exp', '4', 'von_mises_stress', '4']

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from clip1
clip1Display = Show(clip1, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'max_sample'
max_sampleLUT = GetColorTransferFunction('max_sample')
max_sampleLUT.RGBPoints = [-1.2167023438796654, 0.231373, 0.298039, 0.752941, 9.394151289328688, 0.865003, 0.865003, 0.865003, 20.005004922537044, 0.705882, 0.0156863, 0.14902]
max_sampleLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'max_sample'
max_samplePWF = GetOpacityTransferFunction('max_sample')
max_samplePWF.Points = [-1.2167023438796654, 0.0, 0.5, 0.0, 20.005004922537044, 1.0, 0.5, 0.0]
max_samplePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip1Display.Representation = 'Surface'
clip1Display.ColorArrayName = ['CELLS', 'max_sample']
clip1Display.LookupTable = max_sampleLUT
clip1Display.SelectTCoordArray = 'None'
clip1Display.SelectNormalArray = 'None'
clip1Display.SelectTangentArray = 'None'
clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip1Display.SelectOrientationVectors = 'None'
clip1Display.ScaleFactor = 6.000000000000003
clip1Display.SelectScaleArray = 'st_young_modulus'
clip1Display.GlyphType = 'Arrow'
clip1Display.GlyphTableIndexArray = 'st_young_modulus'
clip1Display.GaussianRadius = 0.3000000000000001
clip1Display.SetScaleArray = [None, '']
clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
clip1Display.OpacityArray = [None, '']
clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
clip1Display.DataAxesGrid = 'GridAxesRepresentation'
clip1Display.PolarAxes = 'PolarAxesRepresentation'
clip1Display.ScalarOpacityFunction = max_samplePWF
clip1Display.ScalarOpacityUnitDistance = 3.5861225402363335
clip1Display.OpacityArrayName = ['CELLS', 'st_young_modulus']

# show data from clip2
clip2Display = Show(clip2, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'med_sample'
med_sampleLUT = GetColorTransferFunction('med_sample')
med_sampleLUT.RGBPoints = [-0.40416783091923014, 0.231373, 0.298039, 0.752941, 9.791190386163029, 0.865003, 0.865003, 0.865003, 19.986548603245286, 0.705882, 0.0156863, 0.14902]
med_sampleLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'med_sample'
med_samplePWF = GetOpacityTransferFunction('med_sample')
med_samplePWF.Points = [-0.40416783091923014, 0.0, 0.5, 0.0, 19.986548603245286, 1.0, 0.5, 0.0]
med_samplePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
clip2Display.Representation = 'Surface'
clip2Display.ColorArrayName = ['CELLS', 'med_sample']
clip2Display.LookupTable = med_sampleLUT
clip2Display.SelectTCoordArray = 'None'
clip2Display.SelectNormalArray = 'None'
clip2Display.SelectTangentArray = 'None'
clip2Display.OSPRayScaleFunction = 'PiecewiseFunction'
clip2Display.SelectOrientationVectors = 'None'
clip2Display.ScaleFactor = 7.0
clip2Display.SelectScaleArray = 'st_young_modulus'
clip2Display.GlyphType = 'Arrow'
clip2Display.GlyphTableIndexArray = 'st_young_modulus'
clip2Display.GaussianRadius = 0.35000000000000003
clip2Display.SetScaleArray = [None, '']
clip2Display.ScaleTransferFunction = 'PiecewiseFunction'
clip2Display.OpacityArray = [None, '']
clip2Display.OpacityTransferFunction = 'PiecewiseFunction'
clip2Display.DataAxesGrid = 'GridAxesRepresentation'
clip2Display.PolarAxes = 'PolarAxesRepresentation'
clip2Display.ScalarOpacityFunction = med_samplePWF
clip2Display.ScalarOpacityUnitDistance = 3.449051508050374
clip2Display.OpacityArrayName = ['CELLS', 'st_young_modulus']

# setup the color legend parameters for each legend in this view

# get color legend/bar for max_sampleLUT in view renderView1
max_sampleLUTColorBar = GetScalarBar(max_sampleLUT, renderView1)
max_sampleLUTColorBar.Title = 'max_sample'
max_sampleLUTColorBar.ComponentTitle = ''

# set color bar visibility
max_sampleLUTColorBar.Visibility = 1

# get color legend/bar for med_sampleLUT in view renderView1
med_sampleLUTColorBar = GetScalarBar(med_sampleLUT, renderView1)
med_sampleLUTColorBar.WindowLocation = 'Upper Right Corner'
med_sampleLUTColorBar.Title = 'med_sample'
med_sampleLUTColorBar.ComponentTitle = ''

# set color bar visibility
med_sampleLUTColorBar.Visibility = 1

# show color legend
clip1Display.SetScalarBarVisibility(renderView1, True)

# show color legend
clip2Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from flowpvd
flowpvdDisplay = Show(flowpvd, renderView2, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'conductivity'
conductivityLUT = GetColorTransferFunction('conductivity')
conductivityLUT.RGBPoints = [1.2422606041621623e-09, 0.231373, 0.298039, 0.752941, 5.5641777572812e-06, 0.865003, 0.865003, 0.865003, 1.1127113253958224e-05, 0.705882, 0.0156863, 0.14902]
conductivityLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'conductivity'
conductivityPWF = GetOpacityTransferFunction('conductivity')
conductivityPWF.Points = [1.2422606041621623e-09, 0.0, 0.5, 0.0, 1.1127113253958224e-05, 1.0, 0.5, 0.0]
conductivityPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
flowpvdDisplay.Representation = 'Surface'
flowpvdDisplay.ColorArrayName = ['CELLS', 'conductivity']
flowpvdDisplay.LookupTable = conductivityLUT
flowpvdDisplay.SelectTCoordArray = 'None'
flowpvdDisplay.SelectNormalArray = 'None'
flowpvdDisplay.SelectTangentArray = 'None'
flowpvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
flowpvdDisplay.SelectOrientationVectors = 'None'
flowpvdDisplay.ScaleFactor = 7.0
flowpvdDisplay.SelectScaleArray = 'None'
flowpvdDisplay.GlyphType = 'Arrow'
flowpvdDisplay.GlyphTableIndexArray = 'None'
flowpvdDisplay.GaussianRadius = 0.35000000000000003
flowpvdDisplay.SetScaleArray = [None, '']
flowpvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
flowpvdDisplay.OpacityArray = [None, '']
flowpvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
flowpvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
flowpvdDisplay.PolarAxes = 'PolarAxesRepresentation'
flowpvdDisplay.ScalarOpacityFunction = conductivityPWF
flowpvdDisplay.ScalarOpacityUnitDistance = 3.494669639293449
flowpvdDisplay.OpacityArrayName = ['CELLS', 'conductivity']

# setup the color legend parameters for each legend in this view

# get color legend/bar for conductivityLUT in view renderView2
conductivityLUTColorBar = GetScalarBar(conductivityLUT, renderView2)
conductivityLUTColorBar.Title = 'conductivity'
conductivityLUTColorBar.ComponentTitle = ''

# set color bar visibility
conductivityLUTColorBar.Visibility = 1

# show color legend
flowpvdDisplay.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView3'
# ----------------------------------------------------------------

# show data from calculator4
calculator4Display = Show(calculator4, renderView3, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'Result'
resultLUT = GetColorTransferFunction('Result')
resultLUT.AutomaticRescaleRangeMode = 'Never'
resultLUT.RGBPoints = [-44836453.94326928, 0.231373, 0.298039, 0.752941, -22418226.97163464, 0.865003, 0.865003, 0.865003, 0.0, 0.705882, 0.0156863, 0.14902]
resultLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'Result'
resultPWF = GetOpacityTransferFunction('Result')
resultPWF.Points = [-44836453.94326928, 0.0, 0.5, 0.0, 0.0, 1.0, 0.5, 0.0]
resultPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
calculator4Display.Representation = 'Surface'
calculator4Display.ColorArrayName = ['CELLS', 'Result']
calculator4Display.LookupTable = resultLUT
calculator4Display.SelectTCoordArray = 'None'
calculator4Display.SelectNormalArray = 'None'
calculator4Display.SelectTangentArray = 'None'
calculator4Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator4Display.SelectOrientationVectors = 'None'
calculator4Display.ScaleFactor = 7.0
calculator4Display.SelectScaleArray = 'Result'
calculator4Display.GlyphType = 'Arrow'
calculator4Display.GlyphTableIndexArray = 'Result'
calculator4Display.GaussianRadius = 0.35000000000000003
calculator4Display.SetScaleArray = [None, '']
calculator4Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator4Display.OpacityArray = [None, '']
calculator4Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator4Display.DataAxesGrid = 'GridAxesRepresentation'
calculator4Display.PolarAxes = 'PolarAxesRepresentation'
calculator4Display.ScalarOpacityFunction = resultPWF
calculator4Display.ScalarOpacityUnitDistance = 3.494669639293449
calculator4Display.OpacityArrayName = ['CELLS', 'Result']

# setup the color legend parameters for each legend in this view

# get color legend/bar for resultLUT in view renderView3
resultLUTColorBar = GetScalarBar(resultLUT, renderView3)
resultLUTColorBar.Title = 'Result'
resultLUTColorBar.ComponentTitle = ''

# set color bar visibility
resultLUTColorBar.Visibility = 1

# show color legend
calculator4Display.SetScalarBarVisibility(renderView3, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(calculator2)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')