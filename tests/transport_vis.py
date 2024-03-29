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
renderView1.ViewSize = [1007, 793]
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [60.91750907897949, 7.466259002685547, -2.624338150024414]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [91.33998453014543, -412.17779903930483, -43.10951734777788]
renderView1.CameraFocalPoint = [60.91750907897921, 7.466259002685629, -2.6243381500244953]
renderView1.CameraViewUp = [-0.13474988136439484, 0.08547005457521814, -0.9871865777264148]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 109.39987808307224
renderView1.BackEnd = 'OSPRay raycaster'
renderView1.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [640, 480]
renderView2.AxesGrid = 'GridAxes3DActor'
renderView2.CenterOfRotation = [40.0, 0.0, 0.0]
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [35.29182727459769, 223.39953099899049, -28.746715324633143]
renderView2.CameraFocalPoint = [40.0, 0.0, 0.0]
renderView2.CameraViewUp = [-0.03900072833626413, -0.12833761960602535, -0.9909633689411028]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 58.309518948453004
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.OSPRayMaterialLibrary = materialLibrary1

# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.ViewSize = [640, 480]
renderView3.InteractionMode = '2D'
renderView3.AxesGrid = 'GridAxes3DActor'
renderView3.CenterOfRotation = [40.0, -3.552713678800501e-15, 0.0]
renderView3.StereoType = 'Crystal Eyes'
renderView3.CameraPosition = [40.42941842874936, -359.7797976320743, -4.911030598294696]
renderView3.CameraFocalPoint = [40.42941842874936, -3.552713678800501e-15, -4.911030598294696]
renderView3.CameraViewUp = [0.0, 0.0, 1.0]
renderView3.CameraFocalDisk = 1.0
renderView3.CameraParallelScale = 64.56012095671461
renderView3.BackEnd = 'OSPRay raycaster'
renderView3.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView3.AxesGrid.Visibility = 1

# Create a new 'Render View'
renderView4 = CreateView('RenderView')
renderView4.ViewSize = [640, 480]
renderView4.AxesGrid = 'GridAxes3DActor'
renderView4.CenterOfRotation = [39.99999999999995, -3.552713678800501e-15, 0.0]
renderView4.StereoType = 'Crystal Eyes'
renderView4.CameraPosition = [273.64379146419355, -79.90453293851986, 52.35815555350894]
renderView4.CameraFocalPoint = [-16.42227607199717, 35.446378324256806, 14.47422549864477]
renderView4.CameraViewUp = [-0.13697021209196114, -0.01916372994754449, 0.9903897780439683]
renderView4.CameraFocalDisk = 1.0
renderView4.CameraParallelScale = 82.09892788886275
renderView4.BackEnd = 'OSPRay raycaster'
renderView4.OSPRayMaterialLibrary = materialLibrary1

# init the 'GridAxes3DActor' selected for 'AxesGrid'
renderView4.AxesGrid.Visibility = 1

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Concentration'
concentration = CreateLayout(name='Concentration')
concentration.AssignView(0, renderView1)
concentration.SetSize(1007, 793)

# create new layout object 'SÃ­Å¥ - geometrie'
sgeometrie = CreateLayout(name='SÃ­Å¥ - geometrie')
sgeometrie.SplitHorizontal(0, 0.500000)
sgeometrie.AssignView(1, renderView3)
sgeometrie.AssignView(2, renderView4)
sgeometrie.SetSize(1281, 480)

# create new layout object 'Velocity'
velocity = CreateLayout(name='Velocity')
velocity.AssignView(0, renderView2)
velocity.SetSize(640, 480)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PVD Reader'
flow_fieldspvd = PVDReader(registrationName='flow_fields.pvd', FileName='flow_fields.pvd')
flow_fieldspvd.CellArrays = ['region_id', 'piezo_head_p0', 'velocity_p0', 'cross_section', 'conductivity']

# create a new 'PVD Reader'
solute_fieldspvd = PVDReader(registrationName='solute_fields.pvd', FileName='solute_fields.pvd')
solute_fieldspvd.CellArrays = ['U235_conc']
solute_fieldspvd.PointArrays = ['U235_conc', 'region_id']

# create a new 'Append Attributes'
mergedoutputs = AppendAttributes(registrationName='Merged outputs', Input=[solute_fieldspvd, flow_fieldspvd])

# create a new 'Threshold'
concplume = Threshold(registrationName='Conc plume', Input=mergedoutputs)
concplume.Scalars = ['CELLS', 'I_conc']
concplume.LowerThreshold = 1e-10
concplume.UpperThreshold = 2.4505797548269794

# create a new 'Slice'
topIndicatorSlice = Slice(registrationName='TopIndicatorSlice', Input=mergedoutputs)
topIndicatorSlice.SliceType = 'Plane'
topIndicatorSlice.HyperTreeGridSlicer = 'Plane'
topIndicatorSlice.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
topIndicatorSlice.SliceType.Origin = [39.99999999999995, -3.552713678800501e-15, 27.0]
topIndicatorSlice.SliceType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
topIndicatorSlice.HyperTreeGridSlicer.Origin = [39.99999999999995, -3.552713678800501e-15, 0.0]

# create a new 'Cell Centers'
cellCenters2 = CellCenters(registrationName='CellCenters2', Input=mergedoutputs)

# create a new 'Slice'
slice_Y = Slice(registrationName='Slice_Y', Input=mergedoutputs)
slice_Y.SliceType = 'Plane'
slice_Y.HyperTreeGridSlicer = 'Plane'
slice_Y.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice_Y.SliceType.Origin = [39.99999999999997, 0.0, 0.0]
slice_Y.SliceType.Normal = [0.0, 1.0, 0.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice_Y.HyperTreeGridSlicer.Origin = [39.99999999999997, 0.0, 0.0]

# create a new 'Cell Centers'
cellCenters1 = CellCenters(registrationName='CellCenters1', Input=slice_Y)

# create a new 'Extract Selection'
fractures = ExtractSelection(registrationName='Fractures', Input=mergedoutputs)

# create a new 'Cell Centers'
cellCenters3 = CellCenters(registrationName='CellCenters3', Input=fractures)

# create a new 'Threshold'
macroFr = Threshold(registrationName='MacroFr', Input=fractures)
macroFr.Scalars = ['CELLS', 'region_id']
macroFr.LowerThreshold = 41.0
macroFr.UpperThreshold = 41.0

# create a new 'Threshold'
microFr = Threshold(registrationName='MicroFr', Input=fractures)
microFr.Scalars = ['CELLS', 'region_id']
microFr.LowerThreshold = 39.0
microFr.UpperThreshold = 40.1

# create a new 'Glyph'
slice_Y_Velocity = Glyph(registrationName='Slice_Y_ Velocity', Input=cellCenters1,
    GlyphType='Arrow')
slice_Y_Velocity.OrientationArray = ['POINTS', 'velocity_p0']
slice_Y_Velocity.ScaleArray = ['POINTS', 'No scale array']
slice_Y_Velocity.ScaleFactor = 2.8981389216363844
slice_Y_Velocity.GlyphTransform = 'Transform2'

# create a new 'Glyph'
frVelocity = Glyph(registrationName='FrVelocity', Input=cellCenters3,
    GlyphType='Arrow')
frVelocity.OrientationArray = ['POINTS', 'velocity_p0']
frVelocity.ScaleArray = ['POINTS', 'velocity_p0']
frVelocity.ScaleFactor = 100000.0
frVelocity.GlyphTransform = 'Transform2'
frVelocity.MaximumNumberOfSamplePoints = 80000

# create a new 'Glyph'
bulkVelocity = Glyph(registrationName='BulkVelocity', Input=cellCenters2,
    GlyphType='Arrow')
bulkVelocity.OrientationArray = ['POINTS', 'velocity_p0']
bulkVelocity.ScaleArray = ['POINTS', 'No scale array']
bulkVelocity.ScaleFactor = 5.188808463148893
bulkVelocity.GlyphTransform = 'Transform2'

# create a new 'Slice'
botIndicatorSlice = Slice(registrationName='BotIndicatorSlice', Input=mergedoutputs)
botIndicatorSlice.SliceType = 'Plane'
botIndicatorSlice.HyperTreeGridSlicer = 'Plane'
botIndicatorSlice.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
botIndicatorSlice.SliceType.Origin = [39.99999999999995, -3.552713678800501e-15, -27.0]
botIndicatorSlice.SliceType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
botIndicatorSlice.HyperTreeGridSlicer.Origin = [39.99999999999995, -3.552713678800501e-15, 0.0]

# create a new 'Group Datasets'
indicatorPlanes = GroupDatasets(registrationName='IndicatorPlanes', Input=[botIndicatorSlice, topIndicatorSlice])
indicatorPlanes.BlockNames = ['BotIndicatorSlice', 'TopIndicatorSlice']

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=indicatorPlanes)
contour1.ContourBy = ['POINTS', 'U235_conc']
contour1.Isosurfaces = [0.1250914917361491, 0.01, 0.016681005372000592, 0.027825594022071243, 0.046415888336127774, 0.0774263682681127, 0.1291549665014884, 0.21544346900318834, 0.3593813663804626, 0.5994842503189409, 1.0]
contour1.PointMergeMethod = 'Uniform Binning'

# create a new 'Slice'
slice_Z = Slice(registrationName='Slice_Z', Input=mergedoutputs)
slice_Z.SliceType = 'Plane'
slice_Z.HyperTreeGridSlicer = 'Plane'
slice_Z.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice_Z.SliceType.Origin = [39.99999999999995, 0.0, 1.0]
slice_Z.SliceType.Normal = [0.0, 0.0, 1.0]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice_Z.HyperTreeGridSlicer.Origin = [39.99999999999995, -3.552713678800501e-15, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from mergedoutputs
mergedoutputsDisplay = Show(mergedoutputs, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'region_id'
region_idLUT = GetColorTransferFunction('region_id')
region_idLUT.RGBPoints = [0.0003611691819136407, 0.231373, 0.298039, 0.752941, 23.425220265014797, 0.865003, 0.865003, 0.865003, 46.85007936084768, 0.705882, 0.0156863, 0.14902]
region_idLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'region_id'
region_idPWF = GetOpacityTransferFunction('region_id')
region_idPWF.Points = [0.0003611691819136407, 0.0, 0.5, 0.0, 46.85007936084768, 1.0, 0.5, 0.0]
region_idPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
mergedoutputsDisplay.Representation = 'Surface'
mergedoutputsDisplay.ColorArrayName = ['POINTS', 'region_id']
mergedoutputsDisplay.LookupTable = region_idLUT
mergedoutputsDisplay.SelectTCoordArray = 'None'
mergedoutputsDisplay.SelectNormalArray = 'None'
mergedoutputsDisplay.SelectTangentArray = 'None'
mergedoutputsDisplay.OSPRayScaleArray = 'U235_conc'
mergedoutputsDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
mergedoutputsDisplay.SelectOrientationVectors = 'None'
mergedoutputsDisplay.ScaleFactor = 10.000000000000007
mergedoutputsDisplay.SelectScaleArray = 'None'
mergedoutputsDisplay.GlyphType = 'Arrow'
mergedoutputsDisplay.GlyphTableIndexArray = 'None'
mergedoutputsDisplay.GaussianRadius = 0.5000000000000003
mergedoutputsDisplay.SetScaleArray = ['POINTS', 'U235_conc']
mergedoutputsDisplay.ScaleTransferFunction = 'PiecewiseFunction'
mergedoutputsDisplay.OpacityArray = ['POINTS', 'U235_conc']
mergedoutputsDisplay.OpacityTransferFunction = 'PiecewiseFunction'
mergedoutputsDisplay.DataAxesGrid = 'GridAxesRepresentation'
mergedoutputsDisplay.PolarAxes = 'PolarAxesRepresentation'
mergedoutputsDisplay.ScalarOpacityFunction = region_idPWF
mergedoutputsDisplay.ScalarOpacityUnitDistance = 3.46309045038446
mergedoutputsDisplay.OpacityArrayName = ['POINTS', 'U235_conc']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
mergedoutputsDisplay.ScaleTransferFunction.Points = [-0.15244128536758617, 0.0, 0.5, 0.0, 0.42914035585473564, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
mergedoutputsDisplay.OpacityTransferFunction.Points = [-0.15244128536758617, 0.0, 0.5, 0.0, 0.42914035585473564, 1.0, 0.5, 0.0]

# show data from indicatorPlanes
indicatorPlanesDisplay = Show(indicatorPlanes, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'I_conc'
i_concLUT = GetColorTransferFunction('I_conc')
i_concLUT.RGBPoints = [-1.5481767038066855e-06, 0.231373, 0.298039, 0.752941, 1.3441029621411702e-06, 0.865003, 0.865003, 0.865003, 4.2363826280890264e-06, 0.705882, 0.0156863, 0.14902]
i_concLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
indicatorPlanesDisplay.Representation = 'Surface'
indicatorPlanesDisplay.ColorArrayName = ['CELLS', 'I_conc']
indicatorPlanesDisplay.LookupTable = i_concLUT
indicatorPlanesDisplay.SelectTCoordArray = 'None'
indicatorPlanesDisplay.SelectNormalArray = 'None'
indicatorPlanesDisplay.SelectTangentArray = 'None'
indicatorPlanesDisplay.OSPRayScaleArray = 'I_conc'
indicatorPlanesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
indicatorPlanesDisplay.SelectOrientationVectors = 'None'
indicatorPlanesDisplay.ScaleFactor = 10.0
indicatorPlanesDisplay.SelectScaleArray = 'I_conc'
indicatorPlanesDisplay.GlyphType = 'Arrow'
indicatorPlanesDisplay.GlyphTableIndexArray = 'I_conc'
indicatorPlanesDisplay.GaussianRadius = 0.5
indicatorPlanesDisplay.SetScaleArray = ['POINTS', 'I_conc']
indicatorPlanesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
indicatorPlanesDisplay.OpacityArray = ['POINTS', 'I_conc']
indicatorPlanesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
indicatorPlanesDisplay.DataAxesGrid = 'GridAxesRepresentation'
indicatorPlanesDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
indicatorPlanesDisplay.ScaleTransferFunction.Points = [-2.0795355145365745e-07, 0.0, 0.5, 0.0, 3.0563701098186984e-06, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
indicatorPlanesDisplay.OpacityTransferFunction.Points = [-2.0795355145365745e-07, 0.0, 0.5, 0.0, 3.0563701098186984e-06, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for region_idLUT in view renderView1
region_idLUTColorBar = GetScalarBar(region_idLUT, renderView1)
region_idLUTColorBar.WindowLocation = 'Lower Left Corner'
region_idLUTColorBar.Title = 'region_id'
region_idLUTColorBar.ComponentTitle = ''

# set color bar visibility
region_idLUTColorBar.Visibility = 1

# get color legend/bar for i_concLUT in view renderView1
i_concLUTColorBar = GetScalarBar(i_concLUT, renderView1)
i_concLUTColorBar.WindowLocation = 'Upper Left Corner'
i_concLUTColorBar.Title = 'I_conc'
i_concLUTColorBar.ComponentTitle = ''

# set color bar visibility
i_concLUTColorBar.Visibility = 1

# show color legend
mergedoutputsDisplay.SetScalarBarVisibility(renderView1, True)

# show color legend
indicatorPlanesDisplay.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from slice_Y
slice_YDisplay = Show(slice_Y, renderView2, 'GeometryRepresentation')

# get color transfer function/color map for 'velocity_p0'
velocity_p0LUT = GetColorTransferFunction('velocity_p0')
velocity_p0LUT.RGBPoints = [5.929019716581325e-19, 0.231373, 0.298039, 0.752941, 0.0004998846933968262, 0.865003, 0.865003, 0.865003, 0.0009997693867936522, 0.705882, 0.0156863, 0.14902]
velocity_p0LUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
slice_YDisplay.Representation = 'Surface'
slice_YDisplay.ColorArrayName = ['CELLS', 'velocity_p0']
slice_YDisplay.LookupTable = velocity_p0LUT
slice_YDisplay.SelectTCoordArray = 'None'
slice_YDisplay.SelectNormalArray = 'None'
slice_YDisplay.SelectTangentArray = 'None'
slice_YDisplay.OSPRayScaleArray = 'U235_conc'
slice_YDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
slice_YDisplay.SelectOrientationVectors = 'None'
slice_YDisplay.ScaleFactor = 10.0
slice_YDisplay.SelectScaleArray = 'None'
slice_YDisplay.GlyphType = 'Arrow'
slice_YDisplay.GlyphTableIndexArray = 'None'
slice_YDisplay.GaussianRadius = 0.5
slice_YDisplay.SetScaleArray = ['POINTS', 'U235_conc']
slice_YDisplay.ScaleTransferFunction = 'PiecewiseFunction'
slice_YDisplay.OpacityArray = ['POINTS', 'U235_conc']
slice_YDisplay.OpacityTransferFunction = 'PiecewiseFunction'
slice_YDisplay.DataAxesGrid = 'GridAxesRepresentation'
slice_YDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice_YDisplay.ScaleTransferFunction.Points = [-0.04086492280445489, 0.0, 0.5, 0.0, 0.3861714820282732, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice_YDisplay.OpacityTransferFunction.Points = [-0.04086492280445489, 0.0, 0.5, 0.0, 0.3861714820282732, 1.0, 0.5, 0.0]

# show data from frVelocity
frVelocityDisplay = Show(frVelocity, renderView2, 'GeometryRepresentation')

# trace defaults for the display properties.
frVelocityDisplay.Representation = 'Surface'
frVelocityDisplay.ColorArrayName = [None, '']
frVelocityDisplay.SelectTCoordArray = 'None'
frVelocityDisplay.SelectNormalArray = 'None'
frVelocityDisplay.SelectTangentArray = 'None'
frVelocityDisplay.OSPRayScaleArray = 'U235_conc'
frVelocityDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
frVelocityDisplay.SelectOrientationVectors = 'None'
frVelocityDisplay.ScaleFactor = 10.46499423980713
frVelocityDisplay.SelectScaleArray = 'None'
frVelocityDisplay.GlyphType = 'Arrow'
frVelocityDisplay.GlyphTableIndexArray = 'None'
frVelocityDisplay.GaussianRadius = 0.5232497119903564
frVelocityDisplay.SetScaleArray = ['POINTS', 'U235_conc']
frVelocityDisplay.ScaleTransferFunction = 'PiecewiseFunction'
frVelocityDisplay.OpacityArray = ['POINTS', 'U235_conc']
frVelocityDisplay.OpacityTransferFunction = 'PiecewiseFunction'
frVelocityDisplay.DataAxesGrid = 'GridAxesRepresentation'
frVelocityDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
frVelocityDisplay.ScaleTransferFunction.Points = [-8.961266032696405e-14, 0.0, 0.5, 0.0, 1.1391905101940698e-07, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
frVelocityDisplay.OpacityTransferFunction.Points = [-8.961266032696405e-14, 0.0, 0.5, 0.0, 1.1391905101940698e-07, 1.0, 0.5, 0.0]

# show data from slice_Y_Velocity
slice_Y_VelocityDisplay = Show(slice_Y_Velocity, renderView2, 'GeometryRepresentation')

# trace defaults for the display properties.
slice_Y_VelocityDisplay.Representation = 'Surface'
slice_Y_VelocityDisplay.ColorArrayName = [None, '']
slice_Y_VelocityDisplay.SelectTCoordArray = 'None'
slice_Y_VelocityDisplay.SelectNormalArray = 'None'
slice_Y_VelocityDisplay.SelectTangentArray = 'None'
slice_Y_VelocityDisplay.OSPRayScaleArray = 'U235_conc'
slice_Y_VelocityDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
slice_Y_VelocityDisplay.SelectOrientationVectors = 'None'
slice_Y_VelocityDisplay.ScaleFactor = 10.443181037902832
slice_Y_VelocityDisplay.SelectScaleArray = 'None'
slice_Y_VelocityDisplay.GlyphType = 'Arrow'
slice_Y_VelocityDisplay.GlyphTableIndexArray = 'None'
slice_Y_VelocityDisplay.GaussianRadius = 0.5221590518951416
slice_Y_VelocityDisplay.SetScaleArray = ['POINTS', 'U235_conc']
slice_Y_VelocityDisplay.ScaleTransferFunction = 'PiecewiseFunction'
slice_Y_VelocityDisplay.OpacityArray = ['POINTS', 'U235_conc']
slice_Y_VelocityDisplay.OpacityTransferFunction = 'PiecewiseFunction'
slice_Y_VelocityDisplay.DataAxesGrid = 'GridAxesRepresentation'
slice_Y_VelocityDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice_Y_VelocityDisplay.ScaleTransferFunction.Points = [-0.018388965196310236, 0.0, 0.5, 0.0, 0.24847003508792564, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice_Y_VelocityDisplay.OpacityTransferFunction.Points = [-0.018388965196310236, 0.0, 0.5, 0.0, 0.24847003508792564, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for velocity_p0LUT in view renderView2
velocity_p0LUTColorBar = GetScalarBar(velocity_p0LUT, renderView2)
velocity_p0LUTColorBar.Title = 'velocity_p0'
velocity_p0LUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
velocity_p0LUTColorBar.Visibility = 1

# show color legend
slice_YDisplay.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView3'
# ----------------------------------------------------------------

# show data from macroFr
macroFrDisplay = Show(macroFr, renderView3, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'U235_conc'
u235_concLUT = GetColorTransferFunction('U235_conc')
u235_concLUT.RGBPoints = [0.0004188947050862785, 0.231373, 0.298039, 0.752941, 0.041889470508627846, 0.865003, 0.865003, 0.865003, 4.188947050862785, 0.705882, 0.0156863, 0.14902]
u235_concLUT.UseLogScale = 1
u235_concLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'U235_conc'
u235_concPWF = GetOpacityTransferFunction('U235_conc')
u235_concPWF.Points = [-0.033120488635322756, 0.0, 0.5, 0.0, 4.188947050862783, 1.0, 0.5, 0.0]
u235_concPWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
macroFrDisplay.Representation = 'Wireframe'
macroFrDisplay.AmbientColor = [0.9882352941176471, 1.0, 0.15294117647058825]
macroFrDisplay.ColorArrayName = ['CELLS', 'U235_conc']
macroFrDisplay.DiffuseColor = [0.9882352941176471, 1.0, 0.15294117647058825]
macroFrDisplay.LookupTable = u235_concLUT
macroFrDisplay.LineWidth = 1.5
macroFrDisplay.Interpolation = 'Flat'
macroFrDisplay.Specular = 0.6
macroFrDisplay.Luminosity = 100.0
macroFrDisplay.Diffuse = 0.64
macroFrDisplay.Roughness = 0.72
macroFrDisplay.BaseIOR = 1.0
macroFrDisplay.CoatIOR = 1.93
macroFrDisplay.SelectTCoordArray = 'None'
macroFrDisplay.SelectNormalArray = 'None'
macroFrDisplay.SelectTangentArray = 'None'
macroFrDisplay.OSPRayScaleArray = 'U235_conc'
macroFrDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
macroFrDisplay.SelectOrientationVectors = 'None'
macroFrDisplay.ScaleFactor = 10.0
macroFrDisplay.SelectScaleArray = 'None'
macroFrDisplay.GlyphType = 'Arrow'
macroFrDisplay.GlyphTableIndexArray = 'None'
macroFrDisplay.GaussianRadius = 0.5
macroFrDisplay.SetScaleArray = ['POINTS', 'U235_conc']
macroFrDisplay.ScaleTransferFunction = 'PiecewiseFunction'
macroFrDisplay.OpacityArray = ['POINTS', 'U235_conc']
macroFrDisplay.OpacityTransferFunction = 'PiecewiseFunction'
macroFrDisplay.DataAxesGrid = 'GridAxesRepresentation'
macroFrDisplay.PolarAxes = 'PolarAxesRepresentation'
macroFrDisplay.ScalarOpacityFunction = u235_concPWF
macroFrDisplay.ScalarOpacityUnitDistance = 12.043869697745176
macroFrDisplay.OpacityArrayName = ['POINTS', 'U235_conc']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
macroFrDisplay.ScaleTransferFunction.Points = [-0.14187468122059826, 0.0, 0.5, 0.0, 0.8011779554547451, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
macroFrDisplay.OpacityTransferFunction.Points = [-0.14187468122059826, 0.0, 0.5, 0.0, 0.8011779554547451, 1.0, 0.5, 0.0]

# show data from slice_Y
slice_YDisplay_1 = Show(slice_Y, renderView3, 'GeometryRepresentation')

# trace defaults for the display properties.
slice_YDisplay_1.Representation = 'Surface'
slice_YDisplay_1.ColorArrayName = ['CELLS', 'U235_conc']
slice_YDisplay_1.LookupTable = u235_concLUT
slice_YDisplay_1.SelectTCoordArray = 'None'
slice_YDisplay_1.SelectNormalArray = 'None'
slice_YDisplay_1.SelectTangentArray = 'None'
slice_YDisplay_1.OSPRayScaleArray = 'U235_conc'
slice_YDisplay_1.OSPRayScaleFunction = 'PiecewiseFunction'
slice_YDisplay_1.SelectOrientationVectors = 'None'
slice_YDisplay_1.ScaleFactor = 10.0
slice_YDisplay_1.SelectScaleArray = 'None'
slice_YDisplay_1.GlyphType = 'Arrow'
slice_YDisplay_1.GlyphTableIndexArray = 'None'
slice_YDisplay_1.GaussianRadius = 0.5
slice_YDisplay_1.SetScaleArray = ['POINTS', 'U235_conc']
slice_YDisplay_1.ScaleTransferFunction = 'PiecewiseFunction'
slice_YDisplay_1.OpacityArray = ['POINTS', 'U235_conc']
slice_YDisplay_1.OpacityTransferFunction = 'PiecewiseFunction'
slice_YDisplay_1.DataAxesGrid = 'GridAxesRepresentation'
slice_YDisplay_1.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice_YDisplay_1.ScaleTransferFunction.Points = [-0.009985746152872028, 0.0, 0.5, 0.0, 4.3217532213886924, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice_YDisplay_1.OpacityTransferFunction.Points = [-0.009985746152872028, 0.0, 0.5, 0.0, 4.3217532213886924, 1.0, 0.5, 0.0]

# show data from microFr
microFrDisplay = Show(microFr, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
microFrDisplay.Representation = 'Wireframe'
microFrDisplay.AmbientColor = [0.47058823529411764, 1.0, 0.30980392156862746]
microFrDisplay.ColorArrayName = ['CELLS', 'U235_conc']
microFrDisplay.DiffuseColor = [0.47058823529411764, 1.0, 0.30980392156862746]
microFrDisplay.LookupTable = u235_concLUT
microFrDisplay.LineWidth = 1.5
microFrDisplay.SelectTCoordArray = 'None'
microFrDisplay.SelectNormalArray = 'None'
microFrDisplay.SelectTangentArray = 'None'
microFrDisplay.OSPRayScaleArray = 'U235_conc'
microFrDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
microFrDisplay.SelectOrientationVectors = 'None'
microFrDisplay.ScaleFactor = 10.0
microFrDisplay.SelectScaleArray = 'None'
microFrDisplay.GlyphType = 'Arrow'
microFrDisplay.GlyphTableIndexArray = 'None'
microFrDisplay.GaussianRadius = 0.5
microFrDisplay.SetScaleArray = ['POINTS', 'U235_conc']
microFrDisplay.ScaleTransferFunction = 'PiecewiseFunction'
microFrDisplay.OpacityArray = ['POINTS', 'U235_conc']
microFrDisplay.OpacityTransferFunction = 'PiecewiseFunction'
microFrDisplay.DataAxesGrid = 'GridAxesRepresentation'
microFrDisplay.PolarAxes = 'PolarAxesRepresentation'
microFrDisplay.ScalarOpacityFunction = u235_concPWF
microFrDisplay.ScalarOpacityUnitDistance = 9.846255793287197
microFrDisplay.OpacityArrayName = ['POINTS', 'U235_conc']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
microFrDisplay.ScaleTransferFunction.Points = [-0.06488931495898873, 0.0, 0.5, 0.0, 2.88910614644402, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
microFrDisplay.OpacityTransferFunction.Points = [-0.06488931495898873, 0.0, 0.5, 0.0, 2.88910614644402, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for u235_concLUT in view renderView3
u235_concLUTColorBar = GetScalarBar(u235_concLUT, renderView3)
u235_concLUTColorBar.Orientation = 'Horizontal'
u235_concLUTColorBar.WindowLocation = 'Any Location'
u235_concLUTColorBar.Position = [0.05323572474377758, 0.8546298733855517]
u235_concLUTColorBar.Title = 'U235_conc'
u235_concLUTColorBar.ComponentTitle = ''
u235_concLUTColorBar.ScalarBarLength = 0.5232650073206437

# set color bar visibility
u235_concLUTColorBar.Visibility = 1

# show color legend
macroFrDisplay.SetScalarBarVisibility(renderView3, True)

# show color legend
slice_YDisplay_1.SetScalarBarVisibility(renderView3, True)

# show color legend
microFrDisplay.SetScalarBarVisibility(renderView3, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView4'
# ----------------------------------------------------------------

# show data from macroFr
macroFrDisplay_1 = Show(macroFr, renderView4, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
macroFrDisplay_1.Representation = 'Wireframe'
macroFrDisplay_1.AmbientColor = [1.0, 0.9607843137254902, 0.403921568627451]
macroFrDisplay_1.ColorArrayName = ['CELLS', 'U235_conc']
macroFrDisplay_1.DiffuseColor = [1.0, 0.9607843137254902, 0.403921568627451]
macroFrDisplay_1.LookupTable = u235_concLUT
macroFrDisplay_1.LineWidth = 1.5
macroFrDisplay_1.SelectTCoordArray = 'None'
macroFrDisplay_1.SelectNormalArray = 'None'
macroFrDisplay_1.SelectTangentArray = 'None'
macroFrDisplay_1.OSPRayScaleArray = 'U235_conc'
macroFrDisplay_1.OSPRayScaleFunction = 'PiecewiseFunction'
macroFrDisplay_1.SelectOrientationVectors = 'None'
macroFrDisplay_1.ScaleFactor = 10.0
macroFrDisplay_1.SelectScaleArray = 'None'
macroFrDisplay_1.GlyphType = 'Arrow'
macroFrDisplay_1.GlyphTableIndexArray = 'None'
macroFrDisplay_1.GaussianRadius = 0.5
macroFrDisplay_1.SetScaleArray = ['POINTS', 'U235_conc']
macroFrDisplay_1.ScaleTransferFunction = 'PiecewiseFunction'
macroFrDisplay_1.OpacityArray = ['POINTS', 'U235_conc']
macroFrDisplay_1.OpacityTransferFunction = 'PiecewiseFunction'
macroFrDisplay_1.DataAxesGrid = 'GridAxesRepresentation'
macroFrDisplay_1.PolarAxes = 'PolarAxesRepresentation'
macroFrDisplay_1.ScalarOpacityFunction = u235_concPWF
macroFrDisplay_1.ScalarOpacityUnitDistance = 12.043869697745176
macroFrDisplay_1.OpacityArrayName = ['POINTS', 'U235_conc']

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
macroFrDisplay_1.ScaleTransferFunction.Points = [-0.14187468122059826, 0.0, 0.5, 0.0, 0.8011779554547451, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
macroFrDisplay_1.OpacityTransferFunction.Points = [-0.14187468122059826, 0.0, 0.5, 0.0, 0.8011779554547451, 1.0, 0.5, 0.0]

# show data from slice_Z
slice_ZDisplay = Show(slice_Z, renderView4, 'GeometryRepresentation')

# trace defaults for the display properties.
slice_ZDisplay.Representation = 'Surface'
slice_ZDisplay.ColorArrayName = ['CELLS', 'U235_conc']
slice_ZDisplay.LookupTable = u235_concLUT
slice_ZDisplay.SelectTCoordArray = 'None'
slice_ZDisplay.SelectNormalArray = 'None'
slice_ZDisplay.SelectTangentArray = 'None'
slice_ZDisplay.OSPRayScaleArray = 'U235_conc'
slice_ZDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
slice_ZDisplay.SelectOrientationVectors = 'None'
slice_ZDisplay.ScaleFactor = 10.00000000000001
slice_ZDisplay.SelectScaleArray = 'None'
slice_ZDisplay.GlyphType = 'Arrow'
slice_ZDisplay.GlyphTableIndexArray = 'None'
slice_ZDisplay.GaussianRadius = 0.5000000000000006
slice_ZDisplay.SetScaleArray = ['POINTS', 'U235_conc']
slice_ZDisplay.ScaleTransferFunction = 'PiecewiseFunction'
slice_ZDisplay.OpacityArray = ['POINTS', 'U235_conc']
slice_ZDisplay.OpacityTransferFunction = 'PiecewiseFunction'
slice_ZDisplay.DataAxesGrid = 'GridAxesRepresentation'
slice_ZDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice_ZDisplay.ScaleTransferFunction.Points = [-0.05496787763761295, 0.0, 0.5, 0.0, 4.72315804149542, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice_ZDisplay.OpacityTransferFunction.Points = [-0.05496787763761295, 0.0, 0.5, 0.0, 4.72315804149542, 1.0, 0.5, 0.0]

# show data from microFr
microFrDisplay_1 = Show(microFr, renderView4, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
microFrDisplay_1.Representation = 'Wireframe'
microFrDisplay_1.AmbientColor = [0.5450980392156862, 1.0, 0.34901960784313724]
microFrDisplay_1.ColorArrayName = ['CELLS', 'U235_conc']
microFrDisplay_1.DiffuseColor = [0.5450980392156862, 1.0, 0.34901960784313724]
microFrDisplay_1.LookupTable = u235_concLUT
microFrDisplay_1.LineWidth = 1.5
microFrDisplay_1.SelectTCoordArray = 'None'
microFrDisplay_1.SelectNormalArray = 'None'
microFrDisplay_1.SelectTangentArray = 'None'
microFrDisplay_1.OSPRayScaleFunction = 'PiecewiseFunction'
microFrDisplay_1.SelectOrientationVectors = 'None'
microFrDisplay_1.ScaleFactor = -2.0000000000000002e+298
microFrDisplay_1.SelectScaleArray = 'None'
microFrDisplay_1.GlyphType = 'Arrow'
microFrDisplay_1.GlyphTableIndexArray = 'None'
microFrDisplay_1.GaussianRadius = -1e+297
microFrDisplay_1.SetScaleArray = [None, '']
microFrDisplay_1.ScaleTransferFunction = 'PiecewiseFunction'
microFrDisplay_1.OpacityArray = [None, '']
microFrDisplay_1.OpacityTransferFunction = 'PiecewiseFunction'
microFrDisplay_1.DataAxesGrid = 'GridAxesRepresentation'
microFrDisplay_1.PolarAxes = 'PolarAxesRepresentation'
microFrDisplay_1.ScalarOpacityFunction = u235_concPWF
microFrDisplay_1.OpacityArrayName = [None, '']

# show data from topIndicatorSlice
topIndicatorSliceDisplay = Show(topIndicatorSlice, renderView4, 'GeometryRepresentation')

# trace defaults for the display properties.
topIndicatorSliceDisplay.Representation = 'Surface'
topIndicatorSliceDisplay.ColorArrayName = ['POINTS', 'U235_conc']
topIndicatorSliceDisplay.LookupTable = u235_concLUT
topIndicatorSliceDisplay.SelectTCoordArray = 'None'
topIndicatorSliceDisplay.SelectNormalArray = 'None'
topIndicatorSliceDisplay.SelectTangentArray = 'None'
topIndicatorSliceDisplay.OSPRayScaleArray = 'U235_conc'
topIndicatorSliceDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
topIndicatorSliceDisplay.SelectOrientationVectors = 'None'
topIndicatorSliceDisplay.ScaleFactor = 10.000000000000002
topIndicatorSliceDisplay.SelectScaleArray = 'None'
topIndicatorSliceDisplay.GlyphType = 'Arrow'
topIndicatorSliceDisplay.GlyphTableIndexArray = 'None'
topIndicatorSliceDisplay.GaussianRadius = 0.5000000000000001
topIndicatorSliceDisplay.SetScaleArray = ['POINTS', 'U235_conc']
topIndicatorSliceDisplay.ScaleTransferFunction = 'PiecewiseFunction'
topIndicatorSliceDisplay.OpacityArray = ['POINTS', 'U235_conc']
topIndicatorSliceDisplay.OpacityTransferFunction = 'PiecewiseFunction'
topIndicatorSliceDisplay.DataAxesGrid = 'GridAxesRepresentation'
topIndicatorSliceDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
topIndicatorSliceDisplay.ScaleTransferFunction.Points = [-0.033120488635322756, 0.0, 0.5, 0.0, 0.28330347210762097, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
topIndicatorSliceDisplay.OpacityTransferFunction.Points = [-0.033120488635322756, 0.0, 0.5, 0.0, 0.28330347210762097, 1.0, 0.5, 0.0]

# show data from contour1
contour1Display = Show(contour1, renderView4, 'GeometryRepresentation')

# trace defaults for the display properties.
contour1Display.Representation = 'Surface'
contour1Display.AmbientColor = [0.0, 0.0, 0.0]
contour1Display.ColorArrayName = ['POINTS', '']
contour1Display.DiffuseColor = [0.0, 0.0, 0.0]
contour1Display.LineWidth = 1.5
contour1Display.SelectTCoordArray = 'None'
contour1Display.SelectNormalArray = 'None'
contour1Display.SelectTangentArray = 'None'
contour1Display.OSPRayScaleArray = 'U235_conc'
contour1Display.OSPRayScaleFunction = 'PiecewiseFunction'
contour1Display.SelectOrientationVectors = 'None'
contour1Display.ScaleFactor = 0.7349660361853559
contour1Display.SelectScaleArray = 'U235_conc'
contour1Display.GlyphType = 'Arrow'
contour1Display.GlyphTableIndexArray = 'U235_conc'
contour1Display.GaussianRadius = 0.036748301809267796
contour1Display.SetScaleArray = ['POINTS', 'U235_conc']
contour1Display.ScaleTransferFunction = 'PiecewiseFunction'
contour1Display.OpacityArray = ['POINTS', 'U235_conc']
contour1Display.OpacityTransferFunction = 'PiecewiseFunction'
contour1Display.DataAxesGrid = 'GridAxesRepresentation'
contour1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
contour1Display.ScaleTransferFunction.Points = [0.1250914917361491, 0.0, 0.5, 0.0, 0.12512201070785522, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
contour1Display.OpacityTransferFunction.Points = [0.1250914917361491, 0.0, 0.5, 0.0, 0.12512201070785522, 1.0, 0.5, 0.0]

# show data from botIndicatorSlice
botIndicatorSliceDisplay = Show(botIndicatorSlice, renderView4, 'GeometryRepresentation')

# trace defaults for the display properties.
botIndicatorSliceDisplay.Representation = 'Surface'
botIndicatorSliceDisplay.ColorArrayName = [None, '']
botIndicatorSliceDisplay.SelectTCoordArray = 'None'
botIndicatorSliceDisplay.SelectNormalArray = 'None'
botIndicatorSliceDisplay.SelectTangentArray = 'None'
botIndicatorSliceDisplay.OSPRayScaleArray = 'U235_conc'
botIndicatorSliceDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
botIndicatorSliceDisplay.SelectOrientationVectors = 'None'
botIndicatorSliceDisplay.ScaleFactor = 10.000000000000002
botIndicatorSliceDisplay.SelectScaleArray = 'None'
botIndicatorSliceDisplay.GlyphType = 'Arrow'
botIndicatorSliceDisplay.GlyphTableIndexArray = 'None'
botIndicatorSliceDisplay.GaussianRadius = 0.5000000000000001
botIndicatorSliceDisplay.SetScaleArray = ['POINTS', 'U235_conc']
botIndicatorSliceDisplay.ScaleTransferFunction = 'PiecewiseFunction'
botIndicatorSliceDisplay.OpacityArray = ['POINTS', 'U235_conc']
botIndicatorSliceDisplay.OpacityTransferFunction = 'PiecewiseFunction'
botIndicatorSliceDisplay.DataAxesGrid = 'GridAxesRepresentation'
botIndicatorSliceDisplay.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
botIndicatorSliceDisplay.ScaleTransferFunction.Points = [-0.018342102480340454, 0.0, 0.5, 0.0, 0.1341648618140519, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
botIndicatorSliceDisplay.OpacityTransferFunction.Points = [-0.018342102480340454, 0.0, 0.5, 0.0, 0.1341648618140519, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for u235_concLUT in view renderView4
u235_concLUTColorBar_1 = GetScalarBar(u235_concLUT, renderView4)
u235_concLUTColorBar_1.Orientation = 'Horizontal'
u235_concLUTColorBar_1.WindowLocation = 'Any Location'
u235_concLUTColorBar_1.Position = [0.11326500732064429, 0.8591767781892214]
u235_concLUTColorBar_1.Title = 'U235_conc'
u235_concLUTColorBar_1.ComponentTitle = ''
u235_concLUTColorBar_1.ScalarBarLength = 0.4368814055636897

# set color bar visibility
u235_concLUTColorBar_1.Visibility = 1

# show color legend
macroFrDisplay_1.SetScalarBarVisibility(renderView4, True)

# show color legend
slice_ZDisplay.SetScalarBarVisibility(renderView4, True)

# show color legend
microFrDisplay_1.SetScalarBarVisibility(renderView4, True)

# show color legend
topIndicatorSliceDisplay.SetScalarBarVisibility(renderView4, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'I_conc'
i_concPWF = GetOpacityTransferFunction('I_conc')
i_concPWF.Points = [-1.5481767038066855e-06, 0.0, 0.5, 0.0, 4.2363826280890264e-06, 1.0, 0.5, 0.0]
i_concPWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'velocity_p0'
velocity_p0PWF = GetOpacityTransferFunction('velocity_p0')
velocity_p0PWF.Points = [5.929019716581325e-19, 0.0, 0.5, 0.0, 0.0009997693867936513, 1.0, 0.5, 0.0]
velocity_p0PWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# restore active source
SetActiveSource(mergedoutputs)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')