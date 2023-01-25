# state file generated using paraview version 5.10.1

# uncomment the following three lines to ensure this script works in future versions
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()


def show(solute_file, flow_file):
    # ----------------------------------------------------------------
    # setup views used in the visualization
    # ----------------------------------------------------------------

    # get the material library
    materialLibrary1 = GetMaterialLibrary()

    # Create a new 'Render View'
    renderView1 = CreateView('RenderView')
    renderView1.ViewSize = [1376, 913]
    renderView1.AxesGrid = 'GridAxes3DActor'
    renderView1.CenterOfRotation = [40.0, -0.49280240092524963, 0.0]
    renderView1.StereoType = 'Crystal Eyes'
    renderView1.CameraPosition = [-107.28345567794558, -49.328629134305025, 79.59841684903826]
    renderView1.CameraFocalPoint = [91.5852368362671, -2.993835407004846, -15.605818286952609]
    renderView1.CameraViewUp = [-0.4639063196035972, 0.2374718606701981, -0.8534623846553985]
    renderView1.CameraFocalDisk = 1.0
    renderView1.CameraParallelScale = 58.311601368907354
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
    renderView3.ViewSize = [683, 793]
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
    renderView4.ViewSize = [683, 793]
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
    concentration.SetSize(1376, 913)

    # create new layout object 'SÃ­Å¥ - geometrie'
    sgeometrie = CreateLayout(name='SÃ­Å¥ - geometrie')
    sgeometrie.SplitHorizontal(0, 0.500000)
    sgeometrie.AssignView(1, renderView3)
    sgeometrie.AssignView(2, renderView4)
    sgeometrie.SetSize(1367, 793)

    # create new layout object 'Velocity'
    velocity = CreateLayout(name='Velocity')
    velocity.AssignView(0, renderView2)
    velocity.SetSize(640, 480)

    # ----------------------------------------------------------------
    # restore active view
    SetActiveView(renderView4)
    # ----------------------------------------------------------------

    # ----------------------------------------------------------------
    # setup the data processing pipelines
    # ----------------------------------------------------------------

    # create a new 'Plane'
    plane2 = Plane(registrationName='Plane2')
    plane2.Origin = [-20.0, 50.0, -27.0]
    plane2.Point1 = [100.0, 50.0, -27.0]
    plane2.Point2 = [-20.0, -50.0, -27.0]

    # create a new 'PVD Reader'
    solute_fieldspvd = PVDReader(registrationName='solute_fields.pvd', FileName=solute_file)
    solute_fieldspvd.CellArrays = ['U235_conc']
    solute_fieldspvd.PointArrays = ['U235_conc', 'region_id']

    # create a new 'Plane'
    plane1 = Plane(registrationName='Plane1')
    plane1.Origin = [-20.0, 50.0, 27.0]
    plane1.Point1 = [100.0, 50.0, 27.0]
    plane1.Point2 = [-20.0, -50.0, 27.0]

    # create a new 'PVD Reader'
    flow_fieldspvd = PVDReader(registrationName='flow_fields.pvd', FileName=flow_file)
    flow_fieldspvd.CellArrays = ['region_id', 'piezo_head_p0', 'velocity_p0', 'cross_section', 'conductivity']

    # create a new 'Append Attributes'
    mergedoutputs = AppendAttributes(registrationName='Merged outputs', Input=[solute_fieldspvd, flow_fieldspvd])

    # create a new 'Threshold'
    concplume = Threshold(registrationName='Conc plume', Input=mergedoutputs)
    concplume.Scalars = ['POINTS', 'U235_conc']
    concplume.LowerThreshold = 1e-10
    concplume.UpperThreshold = 2.4505797548269794

    # create a new 'Slice'
    slice4 = Slice(registrationName='Slice4', Input=mergedoutputs)
    slice4.SliceType = 'Plane'
    slice4.HyperTreeGridSlicer = 'Plane'
    slice4.SliceOffsetValues = [0.0]

    # init the 'Plane' selected for 'SliceType'
    slice4.SliceType.Origin = [39.99999999999995, -3.552713678800501e-15, -27.0]
    slice4.SliceType.Normal = [0.0, 0.0, 1.0]

    # init the 'Plane' selected for 'HyperTreeGridSlicer'
    slice4.HyperTreeGridSlicer.Origin = [39.99999999999995, -3.552713678800501e-15, 0.0]

    # create a new 'Extract Selection'
    fractures = ExtractSelection(registrationName='Fractures', Input=mergedoutputs)

    # create a new 'Cell Centers'
    cellCenters3 = CellCenters(registrationName='CellCenters3', Input=fractures)

    # create a new 'Glyph'
    glyph3 = Glyph(registrationName='Glyph3', Input=cellCenters3,
        GlyphType='Arrow')
    glyph3.OrientationArray = ['POINTS', 'velocity_p0']
    glyph3.ScaleArray = ['POINTS', 'velocity_p0']
    glyph3.ScaleFactor = 100000.0
    glyph3.GlyphTransform = 'Transform2'
    glyph3.MaximumNumberOfSamplePoints = 80000

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

    # create a new 'Extract Surface'
    extractSurface1 = ExtractSurface(registrationName='ExtractSurface1', Input=mergedoutputs)

    # create a new 'Clip'
    clip1 = Clip(registrationName='Clip1', Input=extractSurface1)
    clip1.ClipType = 'Plane'
    clip1.HyperTreeGridClipper = 'Plane'
    clip1.Scalars = ['POINTS', 'U235_conc']
    clip1.Value = 2.4344843328115426

    # init the 'Plane' selected for 'ClipType'
    clip1.ClipType.Origin = [39.99999999999995, -3.552713678800501e-15, 10.0]
    clip1.ClipType.Normal = [0.0, 0.0, 1.0]

    # init the 'Plane' selected for 'HyperTreeGridClipper'
    clip1.HyperTreeGridClipper.Origin = [39.99999999999995, -3.552713678800501e-15, 0.0]

    # create a new 'Slice'
    slice1 = Slice(registrationName='Slice1', Input=mergedoutputs)
    slice1.SliceType = 'Plane'
    slice1.HyperTreeGridSlicer = 'Plane'
    slice1.SliceOffsetValues = [0.0]

    # init the 'Plane' selected for 'SliceType'
    slice1.SliceType.Origin = [0.0, 0.0, -27.0]
    slice1.SliceType.Normal = [0.0, 0.0, 1.0]

    # init the 'Plane' selected for 'HyperTreeGridSlicer'
    slice1.HyperTreeGridSlicer.Origin = [39.99999999999997, 0.0, 0.0]

    # create a new 'Slice'
    slice3 = Slice(registrationName='Slice3', Input=mergedoutputs)
    slice3.SliceType = 'Plane'
    slice3.HyperTreeGridSlicer = 'Plane'
    slice3.SliceOffsetValues = [0.0]

    # init the 'Plane' selected for 'SliceType'
    slice3.SliceType.Origin = [39.99999999999995, -3.552713678800501e-15, 27.0]
    slice3.SliceType.Normal = [0.0, 0.0, 1.0]

    # init the 'Plane' selected for 'HyperTreeGridSlicer'
    slice3.HyperTreeGridSlicer.Origin = [39.99999999999995, -3.552713678800501e-15, 0.0]

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

    # create a new 'Glyph'
    glyph1 = Glyph(registrationName='Glyph1', Input=cellCenters1,
        GlyphType='Arrow')
    glyph1.OrientationArray = ['POINTS', 'velocity_p0']
    glyph1.ScaleArray = ['POINTS', 'No scale array']
    glyph1.ScaleFactor = 2.8981389216363844
    glyph1.GlyphTransform = 'Transform2'

    # create a new 'Contour'
    contour1 = Contour(registrationName='Contour1', Input=slice3)
    contour1.ContourBy = ['POINTS', 'U235_conc']
    contour1.Isosurfaces = [0.1250914917361491, 0.01, 0.016681005372000592, 0.027825594022071243, 0.046415888336127774, 0.0774263682681127, 0.1291549665014884, 0.21544346900318834, 0.3593813663804626, 0.5994842503189409, 1.0]
    contour1.PointMergeMethod = 'Uniform Binning'

    # create a new 'Slice'
    slice2 = Slice(registrationName='Slice2', Input=mergedoutputs)
    slice2.SliceType = 'Plane'
    slice2.HyperTreeGridSlicer = 'Plane'
    slice2.SliceOffsetValues = [0.0]

    # init the 'Plane' selected for 'SliceType'
    slice2.SliceType.Origin = [39.99999999999997, 0.0, 28.0]
    slice2.SliceType.Normal = [0.0, 0.0, 1.0]

    # init the 'Plane' selected for 'HyperTreeGridSlicer'
    slice2.HyperTreeGridSlicer.Origin = [39.99999999999997, 0.0, 0.0]

    # create a new 'Cell Centers'
    cellCenters2 = CellCenters(registrationName='CellCenters2', Input=mergedoutputs)

    # create a new 'Glyph'
    glyph2 = Glyph(registrationName='Glyph2', Input=cellCenters2,
        GlyphType='Arrow')
    glyph2.OrientationArray = ['POINTS', 'velocity_p0']
    glyph2.ScaleArray = ['POINTS', 'No scale array']
    glyph2.ScaleFactor = 5.188808463148893
    glyph2.GlyphTransform = 'Transform2'

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView1'
    # ----------------------------------------------------------------

    # show data from solute_fieldspvd
    solute_fieldspvdDisplay = Show(solute_fieldspvd, renderView1, 'UnstructuredGridRepresentation')

    # get color transfer function/color map for 'region_id'
    region_idLUT = GetColorTransferFunction('region_id')
    region_idLUT.RGBPoints = [-0.3408103510351968, 0.231373, 0.298039, 0.752941, 20.3335010744824, 0.865003, 0.865003, 0.865003, 41.0078125, 0.705882, 0.0156863, 0.14902]
    region_idLUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'region_id'
    region_idPWF = GetOpacityTransferFunction('region_id')
    region_idPWF.Points = [-0.3408103510351968, 0.0, 0.5, 0.0, 41.0078125, 1.0, 0.5, 0.0]
    region_idPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    solute_fieldspvdDisplay.Representation = 'Surface'
    solute_fieldspvdDisplay.ColorArrayName = ['POINTS', 'region_id']
    solute_fieldspvdDisplay.LookupTable = region_idLUT
    solute_fieldspvdDisplay.SelectTCoordArray = 'None'
    solute_fieldspvdDisplay.SelectNormalArray = 'None'
    solute_fieldspvdDisplay.SelectTangentArray = 'None'
    solute_fieldspvdDisplay.OSPRayScaleArray = 'U235_conc'
    solute_fieldspvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    solute_fieldspvdDisplay.SelectOrientationVectors = 'None'
    solute_fieldspvdDisplay.ScaleFactor = 10.000000000000007
    solute_fieldspvdDisplay.SelectScaleArray = 'None'
    solute_fieldspvdDisplay.GlyphType = 'Arrow'
    solute_fieldspvdDisplay.GlyphTableIndexArray = 'None'
    solute_fieldspvdDisplay.GaussianRadius = 0.5000000000000003
    solute_fieldspvdDisplay.SetScaleArray = ['POINTS', 'U235_conc']
    solute_fieldspvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    solute_fieldspvdDisplay.OpacityArray = ['POINTS', 'U235_conc']
    solute_fieldspvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    solute_fieldspvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
    solute_fieldspvdDisplay.PolarAxes = 'PolarAxesRepresentation'
    solute_fieldspvdDisplay.ScalarOpacityFunction = region_idPWF
    solute_fieldspvdDisplay.ScalarOpacityUnitDistance = 3.46309045038446
    solute_fieldspvdDisplay.OpacityArrayName = ['POINTS', 'U235_conc']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    solute_fieldspvdDisplay.ScaleTransferFunction.Points = [-474416919.97368205, 0.0, 0.5, 0.0, 1397827925.7508326, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    solute_fieldspvdDisplay.OpacityTransferFunction.Points = [-474416919.97368205, 0.0, 0.5, 0.0, 1397827925.7508326, 1.0, 0.5, 0.0]

    # show data from flow_fieldspvd
    flow_fieldspvdDisplay = Show(flow_fieldspvd, renderView1, 'UnstructuredGridRepresentation')

    # get color transfer function/color map for 'velocity_p0'
    velocity_p0LUT = GetColorTransferFunction('velocity_p0')
    velocity_p0LUT.RGBPoints = [1.5029665608589967e-17, 0.231373, 0.298039, 0.752941, 0.0004998846933968336, 0.865003, 0.865003, 0.865003, 0.0009997693867936522, 0.705882, 0.0156863, 0.14902]
    velocity_p0LUT.ScalarRangeInitialized = 1.0

    # get opacity transfer function/opacity map for 'velocity_p0'
    velocity_p0PWF = GetOpacityTransferFunction('velocity_p0')
    velocity_p0PWF.Points = [1.5029665608589967e-17, 0.0, 0.5, 0.0, 0.0009997693867936513, 1.0, 0.5, 0.0]
    velocity_p0PWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    flow_fieldspvdDisplay.Representation = 'Surface'
    flow_fieldspvdDisplay.ColorArrayName = ['CELLS', 'velocity_p0']
    flow_fieldspvdDisplay.LookupTable = velocity_p0LUT
    flow_fieldspvdDisplay.SelectTCoordArray = 'None'
    flow_fieldspvdDisplay.SelectNormalArray = 'None'
    flow_fieldspvdDisplay.SelectTangentArray = 'None'
    flow_fieldspvdDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    flow_fieldspvdDisplay.SelectOrientationVectors = 'None'
    flow_fieldspvdDisplay.ScaleFactor = 10.000000000000007
    flow_fieldspvdDisplay.SelectScaleArray = 'None'
    flow_fieldspvdDisplay.GlyphType = 'Arrow'
    flow_fieldspvdDisplay.GlyphTableIndexArray = 'None'
    flow_fieldspvdDisplay.GaussianRadius = 0.5000000000000003
    flow_fieldspvdDisplay.SetScaleArray = [None, '']
    flow_fieldspvdDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    flow_fieldspvdDisplay.OpacityArray = [None, '']
    flow_fieldspvdDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    flow_fieldspvdDisplay.DataAxesGrid = 'GridAxesRepresentation'
    flow_fieldspvdDisplay.PolarAxes = 'PolarAxesRepresentation'
    flow_fieldspvdDisplay.ScalarOpacityFunction = velocity_p0PWF
    flow_fieldspvdDisplay.ScalarOpacityUnitDistance = 3.46309045038446
    flow_fieldspvdDisplay.OpacityArrayName = ['CELLS', 'conductivity']

    # show data from slice_Y
    slice_YDisplay = Show(slice_Y, renderView1, 'GeometryRepresentation')

    # get color transfer function/color map for 'U235_conc'
    u235_concLUT = GetColorTransferFunction('U235_conc')
    u235_concLUT.RGBPoints = [0.0004188947050862785, 0.231373, 0.298039, 0.752941, 0.041889470508627846, 0.865003, 0.865003, 0.865003, 4.188947050862785, 0.705882, 0.0156863, 0.14902]
    u235_concLUT.UseLogScale = 1
    u235_concLUT.ScalarRangeInitialized = 1.0

    # trace defaults for the display properties.
    slice_YDisplay.Representation = 'Surface'
    slice_YDisplay.ColorArrayName = ['POINTS', 'U235_conc']
    slice_YDisplay.LookupTable = u235_concLUT
    slice_YDisplay.SelectTCoordArray = 'None'
    slice_YDisplay.SelectNormalArray = 'None'
    slice_YDisplay.SelectTangentArray = 'None'
    slice_YDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    slice_YDisplay.SelectOrientationVectors = 'None'
    slice_YDisplay.ScaleFactor = 10.0
    slice_YDisplay.SelectScaleArray = 'None'
    slice_YDisplay.GlyphType = 'Arrow'
    slice_YDisplay.GlyphTableIndexArray = 'None'
    slice_YDisplay.GaussianRadius = 0.5
    slice_YDisplay.SetScaleArray = [None, '']
    slice_YDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    slice_YDisplay.OpacityArray = [None, '']
    slice_YDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    slice_YDisplay.DataAxesGrid = 'GridAxesRepresentation'
    slice_YDisplay.PolarAxes = 'PolarAxesRepresentation'

    # show data from cellCenters1
    cellCenters1Display = Show(cellCenters1, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    cellCenters1Display.Representation = 'Surface'
    cellCenters1Display.ColorArrayName = [None, '']
    cellCenters1Display.SelectTCoordArray = 'None'
    cellCenters1Display.SelectNormalArray = 'None'
    cellCenters1Display.SelectTangentArray = 'None'
    cellCenters1Display.OSPRayScaleArray = 'conductivity'
    cellCenters1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    cellCenters1Display.SelectOrientationVectors = 'None'
    cellCenters1Display.ScaleFactor = 9.993582488401326
    cellCenters1Display.SelectScaleArray = 'None'
    cellCenters1Display.GlyphType = 'Arrow'
    cellCenters1Display.GlyphTableIndexArray = 'None'
    cellCenters1Display.GaussianRadius = 0.4996791244200663
    cellCenters1Display.SetScaleArray = ['POINTS', 'conductivity']
    cellCenters1Display.ScaleTransferFunction = 'PiecewiseFunction'
    cellCenters1Display.OpacityArray = ['POINTS', 'conductivity']
    cellCenters1Display.OpacityTransferFunction = 'PiecewiseFunction'
    cellCenters1Display.DataAxesGrid = 'GridAxesRepresentation'
    cellCenters1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    cellCenters1Display.ScaleTransferFunction.Points = [1e-13, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    cellCenters1Display.OpacityTransferFunction.Points = [1e-13, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # show data from glyph1
    glyph1Display = Show(glyph1, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    glyph1Display.Representation = 'Surface'
    glyph1Display.ColorArrayName = [None, '']
    glyph1Display.SelectTCoordArray = 'None'
    glyph1Display.SelectNormalArray = 'None'
    glyph1Display.SelectTangentArray = 'None'
    glyph1Display.OSPRayScaleArray = 'conductivity'
    glyph1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    glyph1Display.SelectOrientationVectors = 'None'
    glyph1Display.ScaleFactor = 10.977320098876953
    glyph1Display.SelectScaleArray = 'None'
    glyph1Display.GlyphType = 'Arrow'
    glyph1Display.GlyphTableIndexArray = 'None'
    glyph1Display.GaussianRadius = 0.5488660049438476
    glyph1Display.SetScaleArray = ['POINTS', 'conductivity']
    glyph1Display.ScaleTransferFunction = 'PiecewiseFunction'
    glyph1Display.OpacityArray = ['POINTS', 'conductivity']
    glyph1Display.OpacityTransferFunction = 'PiecewiseFunction'
    glyph1Display.DataAxesGrid = 'GridAxesRepresentation'
    glyph1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    glyph1Display.ScaleTransferFunction.Points = [1e-13, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    glyph1Display.OpacityTransferFunction.Points = [1e-13, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # show data from cellCenters2
    cellCenters2Display = Show(cellCenters2, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    cellCenters2Display.Representation = 'Surface'
    cellCenters2Display.ColorArrayName = [None, '']
    cellCenters2Display.SelectTCoordArray = 'None'
    cellCenters2Display.SelectNormalArray = 'None'
    cellCenters2Display.SelectTangentArray = 'None'
    cellCenters2Display.OSPRayScaleArray = 'conductivity'
    cellCenters2Display.OSPRayScaleFunction = 'PiecewiseFunction'
    cellCenters2Display.SelectOrientationVectors = 'None'
    cellCenters2Display.ScaleFactor = 9.97847781374787
    cellCenters2Display.SelectScaleArray = 'None'
    cellCenters2Display.GlyphType = 'Arrow'
    cellCenters2Display.GlyphTableIndexArray = 'None'
    cellCenters2Display.GaussianRadius = 0.4989238906873935
    cellCenters2Display.SetScaleArray = ['POINTS', 'conductivity']
    cellCenters2Display.ScaleTransferFunction = 'PiecewiseFunction'
    cellCenters2Display.OpacityArray = ['POINTS', 'conductivity']
    cellCenters2Display.OpacityTransferFunction = 'PiecewiseFunction'
    cellCenters2Display.DataAxesGrid = 'GridAxesRepresentation'
    cellCenters2Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    cellCenters2Display.ScaleTransferFunction.Points = [1e-13, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    cellCenters2Display.OpacityTransferFunction.Points = [1e-13, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # show data from glyph2
    glyph2Display = Show(glyph2, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    glyph2Display.Representation = 'Surface'
    glyph2Display.ColorArrayName = [None, '']
    glyph2Display.SelectTCoordArray = 'None'
    glyph2Display.SelectNormalArray = 'None'
    glyph2Display.SelectTangentArray = 'None'
    glyph2Display.OSPRayScaleArray = 'conductivity'
    glyph2Display.OSPRayScaleFunction = 'PiecewiseFunction'
    glyph2Display.SelectOrientationVectors = 'None'
    glyph2Display.ScaleFactor = 10.932991409301758
    glyph2Display.SelectScaleArray = 'None'
    glyph2Display.GlyphType = 'Arrow'
    glyph2Display.GlyphTableIndexArray = 'None'
    glyph2Display.GaussianRadius = 0.5466495704650879
    glyph2Display.SetScaleArray = ['POINTS', 'conductivity']
    glyph2Display.ScaleTransferFunction = 'PiecewiseFunction'
    glyph2Display.OpacityArray = ['POINTS', 'conductivity']
    glyph2Display.OpacityTransferFunction = 'PiecewiseFunction'
    glyph2Display.DataAxesGrid = 'GridAxesRepresentation'
    glyph2Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    glyph2Display.ScaleTransferFunction.Points = [1e-13, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    glyph2Display.OpacityTransferFunction.Points = [1e-13, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # show data from fractures
    fracturesDisplay = Show(fractures, renderView1, 'UnstructuredGridRepresentation')

    # get opacity transfer function/opacity map for 'U235_conc'
    u235_concPWF = GetOpacityTransferFunction('U235_conc')
    u235_concPWF.Points = [-0.033120488635322756, 0.0, 0.5, 0.0, 4.188947050862783, 1.0, 0.5, 0.0]
    u235_concPWF.ScalarRangeInitialized = 1

    # trace defaults for the display properties.
    fracturesDisplay.Representation = 'Surface With Edges'
    fracturesDisplay.ColorArrayName = ['POINTS', 'U235_conc']
    fracturesDisplay.LookupTable = u235_concLUT
    fracturesDisplay.SelectTCoordArray = 'None'
    fracturesDisplay.SelectNormalArray = 'None'
    fracturesDisplay.SelectTangentArray = 'None'
    fracturesDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    fracturesDisplay.SelectOrientationVectors = 'None'
    fracturesDisplay.ScaleFactor = 10.0
    fracturesDisplay.SelectScaleArray = 'None'
    fracturesDisplay.GlyphType = 'Arrow'
    fracturesDisplay.GlyphTableIndexArray = 'None'
    fracturesDisplay.GaussianRadius = 0.5
    fracturesDisplay.SetScaleArray = [None, '']
    fracturesDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    fracturesDisplay.OpacityArray = [None, '']
    fracturesDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    fracturesDisplay.DataAxesGrid = 'GridAxesRepresentation'
    fracturesDisplay.PolarAxes = 'PolarAxesRepresentation'
    fracturesDisplay.ScalarOpacityFunction = u235_concPWF
    fracturesDisplay.ScalarOpacityUnitDistance = 10.915121576588454
    fracturesDisplay.OpacityArrayName = ['CELLS', 'conductivity']

    # show data from cellCenters3
    cellCenters3Display = Show(cellCenters3, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    cellCenters3Display.Representation = 'Surface'
    cellCenters3Display.ColorArrayName = [None, '']
    cellCenters3Display.SelectTCoordArray = 'None'
    cellCenters3Display.SelectNormalArray = 'None'
    cellCenters3Display.SelectTangentArray = 'None'
    cellCenters3Display.OSPRayScaleArray = 'conductivity'
    cellCenters3Display.OSPRayScaleFunction = 'PiecewiseFunction'
    cellCenters3Display.SelectOrientationVectors = 'None'
    cellCenters3Display.ScaleFactor = 9.95199469683665
    cellCenters3Display.SelectScaleArray = 'None'
    cellCenters3Display.GlyphType = 'Arrow'
    cellCenters3Display.GlyphTableIndexArray = 'None'
    cellCenters3Display.GaussianRadius = 0.49759973484183245
    cellCenters3Display.SetScaleArray = ['POINTS', 'conductivity']
    cellCenters3Display.ScaleTransferFunction = 'PiecewiseFunction'
    cellCenters3Display.OpacityArray = ['POINTS', 'conductivity']
    cellCenters3Display.OpacityTransferFunction = 'PiecewiseFunction'
    cellCenters3Display.DataAxesGrid = 'GridAxesRepresentation'
    cellCenters3Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    cellCenters3Display.ScaleTransferFunction.Points = [0.02601849341388111, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    cellCenters3Display.OpacityTransferFunction.Points = [0.02601849341388111, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # show data from glyph3
    glyph3Display = Show(glyph3, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    glyph3Display.Representation = 'Surface'
    glyph3Display.ColorArrayName = ['POINTS', 'velocity_p0']
    glyph3Display.LookupTable = velocity_p0LUT
    glyph3Display.SelectTCoordArray = 'None'
    glyph3Display.SelectNormalArray = 'None'
    glyph3Display.SelectTangentArray = 'None'
    glyph3Display.OSPRayScaleArray = 'conductivity'
    glyph3Display.OSPRayScaleFunction = 'PiecewiseFunction'
    glyph3Display.SelectOrientationVectors = 'None'
    glyph3Display.ScaleFactor = 11.423660850524904
    glyph3Display.SelectScaleArray = 'None'
    glyph3Display.GlyphType = 'Arrow'
    glyph3Display.GlyphTableIndexArray = 'None'
    glyph3Display.GaussianRadius = 0.5711830425262451
    glyph3Display.SetScaleArray = ['POINTS', 'conductivity']
    glyph3Display.ScaleTransferFunction = 'PiecewiseFunction'
    glyph3Display.OpacityArray = ['POINTS', 'conductivity']
    glyph3Display.OpacityTransferFunction = 'PiecewiseFunction'
    glyph3Display.DataAxesGrid = 'GridAxesRepresentation'
    glyph3Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    glyph3Display.ScaleTransferFunction.Points = [0.02601849341388111, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    glyph3Display.OpacityTransferFunction.Points = [0.02601849341388111, 0.0, 0.5, 0.0, 0.7196462202172521, 1.0, 0.5, 0.0]

    # show data from mergedoutputs
    mergedoutputsDisplay = Show(mergedoutputs, renderView1, 'UnstructuredGridRepresentation')

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

    # show data from concplume
    concplumeDisplay = Show(concplume, renderView1, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    concplumeDisplay.Representation = 'Surface'
    concplumeDisplay.ColorArrayName = ['POINTS', 'U235_conc']
    concplumeDisplay.LookupTable = u235_concLUT
    concplumeDisplay.SelectTCoordArray = 'None'
    concplumeDisplay.SelectNormalArray = 'None'
    concplumeDisplay.SelectTangentArray = 'None'
    concplumeDisplay.OSPRayScaleArray = 'U235_conc'
    concplumeDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
    concplumeDisplay.SelectOrientationVectors = 'None'
    concplumeDisplay.ScaleFactor = 7.180571318929471
    concplumeDisplay.SelectScaleArray = 'None'
    concplumeDisplay.GlyphType = 'Arrow'
    concplumeDisplay.GlyphTableIndexArray = 'None'
    concplumeDisplay.GaussianRadius = 0.35902856594647353
    concplumeDisplay.SetScaleArray = ['POINTS', 'U235_conc']
    concplumeDisplay.ScaleTransferFunction = 'PiecewiseFunction'
    concplumeDisplay.OpacityArray = ['POINTS', 'U235_conc']
    concplumeDisplay.OpacityTransferFunction = 'PiecewiseFunction'
    concplumeDisplay.DataAxesGrid = 'GridAxesRepresentation'
    concplumeDisplay.PolarAxes = 'PolarAxesRepresentation'
    concplumeDisplay.ScalarOpacityFunction = u235_concPWF
    concplumeDisplay.ScalarOpacityUnitDistance = 2.278385731082968
    concplumeDisplay.OpacityArrayName = ['POINTS', 'U235_conc']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    concplumeDisplay.ScaleTransferFunction.Points = [0.0010006058395585268, 0.0, 0.5, 0.0, 2.4505797548269794, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    concplumeDisplay.OpacityTransferFunction.Points = [0.0010006058395585268, 0.0, 0.5, 0.0, 2.4505797548269794, 1.0, 0.5, 0.0]

    # show data from slice1
    slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice1Display.Representation = 'Surface'
    slice1Display.ColorArrayName = ['POINTS', 'U235_conc']
    slice1Display.LookupTable = u235_concLUT
    slice1Display.SelectTCoordArray = 'None'
    slice1Display.SelectNormalArray = 'None'
    slice1Display.SelectTangentArray = 'None'
    slice1Display.OSPRayScaleArray = 'U235_conc'
    slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    slice1Display.SelectOrientationVectors = 'None'
    slice1Display.ScaleFactor = 10.000000000000007
    slice1Display.SelectScaleArray = 'None'
    slice1Display.GlyphType = 'Arrow'
    slice1Display.GlyphTableIndexArray = 'None'
    slice1Display.GaussianRadius = 0.5000000000000003
    slice1Display.SetScaleArray = ['POINTS', 'U235_conc']
    slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
    slice1Display.OpacityArray = ['POINTS', 'U235_conc']
    slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
    slice1Display.DataAxesGrid = 'GridAxesRepresentation'
    slice1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    slice1Display.ScaleTransferFunction.Points = [-0.5562685743522303, 0.0, 0.5, 0.0, 0.9347520996003366, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    slice1Display.OpacityTransferFunction.Points = [-0.5562685743522303, 0.0, 0.5, 0.0, 0.9347520996003366, 1.0, 0.5, 0.0]

    # show data from slice2
    slice2Display = Show(slice2, renderView1, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice2Display.Representation = 'Surface'
    slice2Display.ColorArrayName = ['POINTS', 'U235_conc']
    slice2Display.LookupTable = u235_concLUT
    slice2Display.SelectTCoordArray = 'None'
    slice2Display.SelectNormalArray = 'None'
    slice2Display.SelectTangentArray = 'None'
    slice2Display.OSPRayScaleArray = 'U235_conc'
    slice2Display.OSPRayScaleFunction = 'PiecewiseFunction'
    slice2Display.SelectOrientationVectors = 'None'
    slice2Display.ScaleFactor = 10.0
    slice2Display.SelectScaleArray = 'None'
    slice2Display.GlyphType = 'Arrow'
    slice2Display.GlyphTableIndexArray = 'None'
    slice2Display.GaussianRadius = 0.5
    slice2Display.SetScaleArray = ['POINTS', 'U235_conc']
    slice2Display.ScaleTransferFunction = 'PiecewiseFunction'
    slice2Display.OpacityArray = ['POINTS', 'U235_conc']
    slice2Display.OpacityTransferFunction = 'PiecewiseFunction'
    slice2Display.DataAxesGrid = 'GridAxesRepresentation'
    slice2Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    slice2Display.ScaleTransferFunction.Points = [-0.0021582825194322412, 0.0, 0.5, 0.0, 0.002382526139818664, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    slice2Display.OpacityTransferFunction.Points = [-0.0021582825194322412, 0.0, 0.5, 0.0, 0.002382526139818664, 1.0, 0.5, 0.0]

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for u235_concLUT in view renderView1
    u235_concLUTColorBar = GetScalarBar(u235_concLUT, renderView1)
    u235_concLUTColorBar.Title = 'U235_conc'
    u235_concLUTColorBar.ComponentTitle = ''

    # set color bar visibility
    u235_concLUTColorBar.Visibility = 1

    # get color legend/bar for velocity_p0LUT in view renderView1
    velocity_p0LUTColorBar = GetScalarBar(velocity_p0LUT, renderView1)
    velocity_p0LUTColorBar.WindowLocation = 'Upper Right Corner'
    velocity_p0LUTColorBar.Title = 'velocity_p0'
    velocity_p0LUTColorBar.ComponentTitle = 'Magnitude'

    # set color bar visibility
    velocity_p0LUTColorBar.Visibility = 1

    # get color legend/bar for region_idLUT in view renderView1
    region_idLUTColorBar = GetScalarBar(region_idLUT, renderView1)
    region_idLUTColorBar.WindowLocation = 'Upper Left Corner'
    region_idLUTColorBar.Title = 'region_id'
    region_idLUTColorBar.ComponentTitle = ''

    # set color bar visibility
    region_idLUTColorBar.Visibility = 1

    # show color legend
    solute_fieldspvdDisplay.SetScalarBarVisibility(renderView1, True)

    # hide data in view
    Hide(solute_fieldspvd, renderView1)

    # show color legend
    flow_fieldspvdDisplay.SetScalarBarVisibility(renderView1, True)

    # hide data in view
    Hide(flow_fieldspvd, renderView1)

    # show color legend
    slice_YDisplay.SetScalarBarVisibility(renderView1, True)

    # hide data in view
    Hide(slice_Y, renderView1)

    # hide data in view
    Hide(cellCenters1, renderView1)

    # hide data in view
    Hide(glyph1, renderView1)

    # hide data in view
    Hide(cellCenters2, renderView1)

    # hide data in view
    Hide(glyph2, renderView1)

    # show color legend
    fracturesDisplay.SetScalarBarVisibility(renderView1, True)

    # hide data in view
    Hide(cellCenters3, renderView1)

    # show color legend
    glyph3Display.SetScalarBarVisibility(renderView1, True)

    # show color legend
    mergedoutputsDisplay.SetScalarBarVisibility(renderView1, True)

    # show color legend
    concplumeDisplay.SetScalarBarVisibility(renderView1, True)

    # hide data in view
    Hide(concplume, renderView1)

    # show color legend
    slice1Display.SetScalarBarVisibility(renderView1, True)

    # hide data in view
    Hide(slice1, renderView1)

    # show color legend
    slice2Display.SetScalarBarVisibility(renderView1, True)

    # hide data in view
    Hide(slice2, renderView1)

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView2'
    # ----------------------------------------------------------------

    # show data from slice_Y
    slice_YDisplay_1 = Show(slice_Y, renderView2, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice_YDisplay_1.Representation = 'Surface'
    slice_YDisplay_1.ColorArrayName = ['CELLS', 'velocity_p0']
    slice_YDisplay_1.LookupTable = velocity_p0LUT
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
    slice_YDisplay_1.ScaleTransferFunction.Points = [-0.04086492280445489, 0.0, 0.5, 0.0, 0.3861714820282732, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    slice_YDisplay_1.OpacityTransferFunction.Points = [-0.04086492280445489, 0.0, 0.5, 0.0, 0.3861714820282732, 1.0, 0.5, 0.0]

    # show data from glyph3
    glyph3Display_1 = Show(glyph3, renderView2, 'GeometryRepresentation')

    # trace defaults for the display properties.
    glyph3Display_1.Representation = 'Surface'
    glyph3Display_1.ColorArrayName = [None, '']
    glyph3Display_1.SelectTCoordArray = 'None'
    glyph3Display_1.SelectNormalArray = 'None'
    glyph3Display_1.SelectTangentArray = 'None'
    glyph3Display_1.OSPRayScaleArray = 'U235_conc'
    glyph3Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
    glyph3Display_1.SelectOrientationVectors = 'None'
    glyph3Display_1.ScaleFactor = 10.46499423980713
    glyph3Display_1.SelectScaleArray = 'None'
    glyph3Display_1.GlyphType = 'Arrow'
    glyph3Display_1.GlyphTableIndexArray = 'None'
    glyph3Display_1.GaussianRadius = 0.5232497119903564
    glyph3Display_1.SetScaleArray = ['POINTS', 'U235_conc']
    glyph3Display_1.ScaleTransferFunction = 'PiecewiseFunction'
    glyph3Display_1.OpacityArray = ['POINTS', 'U235_conc']
    glyph3Display_1.OpacityTransferFunction = 'PiecewiseFunction'
    glyph3Display_1.DataAxesGrid = 'GridAxesRepresentation'
    glyph3Display_1.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    glyph3Display_1.ScaleTransferFunction.Points = [-8.961266032696405e-14, 0.0, 0.5, 0.0, 1.1391905101940698e-07, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    glyph3Display_1.OpacityTransferFunction.Points = [-8.961266032696405e-14, 0.0, 0.5, 0.0, 1.1391905101940698e-07, 1.0, 0.5, 0.0]

    # show data from glyph2
    glyph2Display_1 = Show(glyph2, renderView2, 'GeometryRepresentation')

    # trace defaults for the display properties.
    glyph2Display_1.Representation = 'Surface'
    glyph2Display_1.ColorArrayName = [None, '']
    glyph2Display_1.SelectTCoordArray = 'None'
    glyph2Display_1.SelectNormalArray = 'None'
    glyph2Display_1.SelectTangentArray = 'None'
    glyph2Display_1.OSPRayScaleArray = 'U235_conc'
    glyph2Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
    glyph2Display_1.SelectOrientationVectors = 'None'
    glyph2Display_1.ScaleFactor = 10.807801914215089
    glyph2Display_1.SelectScaleArray = 'None'
    glyph2Display_1.GlyphType = 'Arrow'
    glyph2Display_1.GlyphTableIndexArray = 'None'
    glyph2Display_1.GaussianRadius = 0.5403900957107544
    glyph2Display_1.SetScaleArray = ['POINTS', 'U235_conc']
    glyph2Display_1.ScaleTransferFunction = 'PiecewiseFunction'
    glyph2Display_1.OpacityArray = ['POINTS', 'U235_conc']
    glyph2Display_1.OpacityTransferFunction = 'PiecewiseFunction'
    glyph2Display_1.DataAxesGrid = 'GridAxesRepresentation'
    glyph2Display_1.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    glyph2Display_1.ScaleTransferFunction.Points = [-3.831731015104576e-05, 0.0, 0.5, 0.0, 0.005957556160557615, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    glyph2Display_1.OpacityTransferFunction.Points = [-3.831731015104576e-05, 0.0, 0.5, 0.0, 0.005957556160557615, 1.0, 0.5, 0.0]

    # show data from glyph1
    glyph1Display_1 = Show(glyph1, renderView2, 'GeometryRepresentation')

    # trace defaults for the display properties.
    glyph1Display_1.Representation = 'Surface'
    glyph1Display_1.ColorArrayName = [None, '']
    glyph1Display_1.SelectTCoordArray = 'None'
    glyph1Display_1.SelectNormalArray = 'None'
    glyph1Display_1.SelectTangentArray = 'None'
    glyph1Display_1.OSPRayScaleArray = 'U235_conc'
    glyph1Display_1.OSPRayScaleFunction = 'PiecewiseFunction'
    glyph1Display_1.SelectOrientationVectors = 'None'
    glyph1Display_1.ScaleFactor = 10.443181037902832
    glyph1Display_1.SelectScaleArray = 'None'
    glyph1Display_1.GlyphType = 'Arrow'
    glyph1Display_1.GlyphTableIndexArray = 'None'
    glyph1Display_1.GaussianRadius = 0.5221590518951416
    glyph1Display_1.SetScaleArray = ['POINTS', 'U235_conc']
    glyph1Display_1.ScaleTransferFunction = 'PiecewiseFunction'
    glyph1Display_1.OpacityArray = ['POINTS', 'U235_conc']
    glyph1Display_1.OpacityTransferFunction = 'PiecewiseFunction'
    glyph1Display_1.DataAxesGrid = 'GridAxesRepresentation'
    glyph1Display_1.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    glyph1Display_1.ScaleTransferFunction.Points = [-0.018388965196310236, 0.0, 0.5, 0.0, 0.24847003508792564, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    glyph1Display_1.OpacityTransferFunction.Points = [-0.018388965196310236, 0.0, 0.5, 0.0, 0.24847003508792564, 1.0, 0.5, 0.0]

    # show data from fractures
    fracturesDisplay_1 = Show(fractures, renderView2, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    fracturesDisplay_1.Representation = 'Surface'
    fracturesDisplay_1.ColorArrayName = ['CELLS', 'velocity_p0']
    fracturesDisplay_1.LookupTable = velocity_p0LUT
    fracturesDisplay_1.SelectTCoordArray = 'None'
    fracturesDisplay_1.SelectNormalArray = 'None'
    fracturesDisplay_1.SelectTangentArray = 'None'
    fracturesDisplay_1.OSPRayScaleArray = 'U235_conc'
    fracturesDisplay_1.OSPRayScaleFunction = 'PiecewiseFunction'
    fracturesDisplay_1.SelectOrientationVectors = 'None'
    fracturesDisplay_1.ScaleFactor = 10.0
    fracturesDisplay_1.SelectScaleArray = 'None'
    fracturesDisplay_1.GlyphType = 'Arrow'
    fracturesDisplay_1.GlyphTableIndexArray = 'None'
    fracturesDisplay_1.GaussianRadius = 0.5
    fracturesDisplay_1.SetScaleArray = ['POINTS', 'U235_conc']
    fracturesDisplay_1.ScaleTransferFunction = 'PiecewiseFunction'
    fracturesDisplay_1.OpacityArray = ['POINTS', 'U235_conc']
    fracturesDisplay_1.OpacityTransferFunction = 'PiecewiseFunction'
    fracturesDisplay_1.DataAxesGrid = 'GridAxesRepresentation'
    fracturesDisplay_1.PolarAxes = 'PolarAxesRepresentation'
    fracturesDisplay_1.ScalarOpacityFunction = velocity_p0PWF
    fracturesDisplay_1.ScalarOpacityUnitDistance = 10.915121576588454
    fracturesDisplay_1.OpacityArrayName = ['POINTS', 'U235_conc']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fracturesDisplay_1.ScaleTransferFunction.Points = [-1.5522912288650734e-09, 0.0, 0.5, 0.0, 1.4307185201518277e-07, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fracturesDisplay_1.OpacityTransferFunction.Points = [-1.5522912288650734e-09, 0.0, 0.5, 0.0, 1.4307185201518277e-07, 1.0, 0.5, 0.0]

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for velocity_p0LUT in view renderView2
    velocity_p0LUTColorBar_1 = GetScalarBar(velocity_p0LUT, renderView2)
    velocity_p0LUTColorBar_1.Title = 'velocity_p0'
    velocity_p0LUTColorBar_1.ComponentTitle = 'Magnitude'

    # set color bar visibility
    velocity_p0LUTColorBar_1.Visibility = 1

    # show color legend
    slice_YDisplay_1.SetScalarBarVisibility(renderView2, True)

    # hide data in view
    Hide(glyph2, renderView2)

    # show color legend
    fracturesDisplay_1.SetScalarBarVisibility(renderView2, True)

    # hide data in view
    Hide(fractures, renderView2)

    # ----------------------------------------------------------------
    # setup the visualization in view 'renderView3'
    # ----------------------------------------------------------------

    # show data from mergedoutputs
    mergedoutputsDisplay_1 = Show(mergedoutputs, renderView3, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    mergedoutputsDisplay_1.Representation = 'Surface'
    mergedoutputsDisplay_1.ColorArrayName = [None, '']
    mergedoutputsDisplay_1.SelectTCoordArray = 'None'
    mergedoutputsDisplay_1.SelectNormalArray = 'None'
    mergedoutputsDisplay_1.SelectTangentArray = 'None'
    mergedoutputsDisplay_1.OSPRayScaleArray = 'U235_conc'
    mergedoutputsDisplay_1.OSPRayScaleFunction = 'PiecewiseFunction'
    mergedoutputsDisplay_1.SelectOrientationVectors = 'None'
    mergedoutputsDisplay_1.ScaleFactor = 10.00000000000001
    mergedoutputsDisplay_1.SelectScaleArray = 'None'
    mergedoutputsDisplay_1.GlyphType = 'Arrow'
    mergedoutputsDisplay_1.GlyphTableIndexArray = 'None'
    mergedoutputsDisplay_1.GaussianRadius = 0.5000000000000006
    mergedoutputsDisplay_1.SetScaleArray = ['POINTS', 'U235_conc']
    mergedoutputsDisplay_1.ScaleTransferFunction = 'PiecewiseFunction'
    mergedoutputsDisplay_1.OpacityArray = ['POINTS', 'U235_conc']
    mergedoutputsDisplay_1.OpacityTransferFunction = 'PiecewiseFunction'
    mergedoutputsDisplay_1.DataAxesGrid = 'GridAxesRepresentation'
    mergedoutputsDisplay_1.PolarAxes = 'PolarAxesRepresentation'
    mergedoutputsDisplay_1.ScalarOpacityUnitDistance = 3.1509480300246997
    mergedoutputsDisplay_1.OpacityArrayName = ['POINTS', 'U235_conc']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    mergedoutputsDisplay_1.ScaleTransferFunction.Points = [-0.26518405488851754, 0.0, 0.5, 0.0, 5.134152720511603, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    mergedoutputsDisplay_1.OpacityTransferFunction.Points = [-0.26518405488851754, 0.0, 0.5, 0.0, 5.134152720511603, 1.0, 0.5, 0.0]

    # show data from fractures
    fracturesDisplay_2 = Show(fractures, renderView3, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    fracturesDisplay_2.Representation = 'Surface'
    fracturesDisplay_2.ColorArrayName = ['POINTS', '']
    fracturesDisplay_2.Interpolation = 'Flat'
    fracturesDisplay_2.Specular = 0.59
    fracturesDisplay_2.Luminosity = 100.0
    fracturesDisplay_2.Diffuse = 0.61
    fracturesDisplay_2.SelectTCoordArray = 'None'
    fracturesDisplay_2.SelectNormalArray = 'None'
    fracturesDisplay_2.SelectTangentArray = 'None'
    fracturesDisplay_2.OSPRayScaleArray = 'U235_conc'
    fracturesDisplay_2.OSPRayScaleFunction = 'PiecewiseFunction'
    fracturesDisplay_2.SelectOrientationVectors = 'None'
    fracturesDisplay_2.ScaleFactor = 10.0
    fracturesDisplay_2.SelectScaleArray = 'None'
    fracturesDisplay_2.GlyphType = 'Arrow'
    fracturesDisplay_2.GlyphTableIndexArray = 'None'
    fracturesDisplay_2.GaussianRadius = 0.5
    fracturesDisplay_2.SetScaleArray = ['POINTS', 'U235_conc']
    fracturesDisplay_2.ScaleTransferFunction = 'PiecewiseFunction'
    fracturesDisplay_2.OpacityArray = ['POINTS', 'U235_conc']
    fracturesDisplay_2.OpacityTransferFunction = 'PiecewiseFunction'
    fracturesDisplay_2.DataAxesGrid = 'GridAxesRepresentation'
    fracturesDisplay_2.PolarAxes = 'PolarAxesRepresentation'
    fracturesDisplay_2.ScalarOpacityUnitDistance = 8.514581422671633
    fracturesDisplay_2.OpacityArrayName = ['POINTS', 'U235_conc']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fracturesDisplay_2.ScaleTransferFunction.Points = [-0.14187468122059826, 0.0, 0.5, 0.0, 2.88910614644402, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fracturesDisplay_2.OpacityTransferFunction.Points = [-0.14187468122059826, 0.0, 0.5, 0.0, 2.88910614644402, 1.0, 0.5, 0.0]

    # show data from macroFr
    macroFrDisplay = Show(macroFr, renderView3, 'UnstructuredGridRepresentation')

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
    slice_YDisplay_2 = Show(slice_Y, renderView3, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice_YDisplay_2.Representation = 'Surface'
    slice_YDisplay_2.ColorArrayName = ['CELLS', 'U235_conc']
    slice_YDisplay_2.LookupTable = u235_concLUT
    slice_YDisplay_2.SelectTCoordArray = 'None'
    slice_YDisplay_2.SelectNormalArray = 'None'
    slice_YDisplay_2.SelectTangentArray = 'None'
    slice_YDisplay_2.OSPRayScaleArray = 'U235_conc'
    slice_YDisplay_2.OSPRayScaleFunction = 'PiecewiseFunction'
    slice_YDisplay_2.SelectOrientationVectors = 'None'
    slice_YDisplay_2.ScaleFactor = 10.0
    slice_YDisplay_2.SelectScaleArray = 'None'
    slice_YDisplay_2.GlyphType = 'Arrow'
    slice_YDisplay_2.GlyphTableIndexArray = 'None'
    slice_YDisplay_2.GaussianRadius = 0.5
    slice_YDisplay_2.SetScaleArray = ['POINTS', 'U235_conc']
    slice_YDisplay_2.ScaleTransferFunction = 'PiecewiseFunction'
    slice_YDisplay_2.OpacityArray = ['POINTS', 'U235_conc']
    slice_YDisplay_2.OpacityTransferFunction = 'PiecewiseFunction'
    slice_YDisplay_2.DataAxesGrid = 'GridAxesRepresentation'
    slice_YDisplay_2.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    slice_YDisplay_2.ScaleTransferFunction.Points = [-0.009985746152872028, 0.0, 0.5, 0.0, 4.3217532213886924, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    slice_YDisplay_2.OpacityTransferFunction.Points = [-0.009985746152872028, 0.0, 0.5, 0.0, 4.3217532213886924, 1.0, 0.5, 0.0]

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

    # get color legend/bar for region_idLUT in view renderView3
    region_idLUTColorBar_1 = GetScalarBar(region_idLUT, renderView3)
    region_idLUTColorBar_1.Title = 'region_id'
    region_idLUTColorBar_1.ComponentTitle = ''

    # set color bar visibility
    region_idLUTColorBar_1.Visibility = 0

    # get color legend/bar for u235_concLUT in view renderView3
    u235_concLUTColorBar_1 = GetScalarBar(u235_concLUT, renderView3)
    u235_concLUTColorBar_1.Orientation = 'Horizontal'
    u235_concLUTColorBar_1.WindowLocation = 'Any Location'
    u235_concLUTColorBar_1.Position = [0.05323572474377758, 0.8546298733855517]
    u235_concLUTColorBar_1.Title = 'U235_conc'
    u235_concLUTColorBar_1.ComponentTitle = ''
    u235_concLUTColorBar_1.ScalarBarLength = 0.5232650073206437

    # set color bar visibility
    u235_concLUTColorBar_1.Visibility = 1

    # get color transfer function/color map for 'piezo_head_p0'
    piezo_head_p0LUT = GetColorTransferFunction('piezo_head_p0')
    piezo_head_p0LUT.RGBPoints = [656.4727751933933, 0.231373, 0.298039, 0.752941, 658.2736673342441, 0.865003, 0.865003, 0.865003, 660.0745594750949, 0.705882, 0.0156863, 0.14902]
    piezo_head_p0LUT.ScalarRangeInitialized = 1.0

    # get color legend/bar for piezo_head_p0LUT in view renderView3
    piezo_head_p0LUTColorBar = GetScalarBar(piezo_head_p0LUT, renderView3)
    piezo_head_p0LUTColorBar.Title = 'piezo_head_p0'
    piezo_head_p0LUTColorBar.ComponentTitle = ''

    # set color bar visibility
    piezo_head_p0LUTColorBar.Visibility = 0

    # hide data in view
    Hide(mergedoutputs, renderView3)

    # hide data in view
    Hide(fractures, renderView3)

    # show color legend
    macroFrDisplay.SetScalarBarVisibility(renderView3, True)

    # show color legend
    slice_YDisplay_2.SetScalarBarVisibility(renderView3, True)

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

    # show data from fractures
    fracturesDisplay_3 = Show(fractures, renderView4, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    fracturesDisplay_3.Representation = 'Surface'
    fracturesDisplay_3.AmbientColor = [0.0, 1.0, 0.0]
    fracturesDisplay_3.ColorArrayName = [None, '']
    fracturesDisplay_3.DiffuseColor = [0.0, 1.0, 0.0]
    fracturesDisplay_3.SelectTCoordArray = 'None'
    fracturesDisplay_3.SelectNormalArray = 'None'
    fracturesDisplay_3.SelectTangentArray = 'None'
    fracturesDisplay_3.OSPRayScaleArray = 'U235_conc'
    fracturesDisplay_3.OSPRayScaleFunction = 'PiecewiseFunction'
    fracturesDisplay_3.SelectOrientationVectors = 'None'
    fracturesDisplay_3.ScaleFactor = 10.0
    fracturesDisplay_3.SelectScaleArray = 'None'
    fracturesDisplay_3.GlyphType = 'Arrow'
    fracturesDisplay_3.GlyphTableIndexArray = 'None'
    fracturesDisplay_3.GaussianRadius = 0.5
    fracturesDisplay_3.SetScaleArray = ['POINTS', 'U235_conc']
    fracturesDisplay_3.ScaleTransferFunction = 'PiecewiseFunction'
    fracturesDisplay_3.OpacityArray = ['POINTS', 'U235_conc']
    fracturesDisplay_3.OpacityTransferFunction = 'PiecewiseFunction'
    fracturesDisplay_3.DataAxesGrid = 'GridAxesRepresentation'
    fracturesDisplay_3.PolarAxes = 'PolarAxesRepresentation'
    fracturesDisplay_3.ScalarOpacityUnitDistance = 8.514581422671633
    fracturesDisplay_3.OpacityArrayName = ['POINTS', 'U235_conc']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    fracturesDisplay_3.ScaleTransferFunction.Points = [-0.14187468122059826, 0.0, 0.5, 0.0, 2.88910614644402, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    fracturesDisplay_3.OpacityTransferFunction.Points = [-0.14187468122059826, 0.0, 0.5, 0.0, 2.88910614644402, 1.0, 0.5, 0.0]

    # show data from slice_Y
    slice_YDisplay_3 = Show(slice_Y, renderView4, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice_YDisplay_3.Representation = 'Surface'
    slice_YDisplay_3.ColorArrayName = ['CELLS', 'U235_conc']
    slice_YDisplay_3.LookupTable = u235_concLUT
    slice_YDisplay_3.SelectTCoordArray = 'None'
    slice_YDisplay_3.SelectNormalArray = 'None'
    slice_YDisplay_3.SelectTangentArray = 'None'
    slice_YDisplay_3.OSPRayScaleArray = 'U235_conc'
    slice_YDisplay_3.OSPRayScaleFunction = 'PiecewiseFunction'
    slice_YDisplay_3.SelectOrientationVectors = 'None'
    slice_YDisplay_3.ScaleFactor = 10.0
    slice_YDisplay_3.SelectScaleArray = 'None'
    slice_YDisplay_3.GlyphType = 'Arrow'
    slice_YDisplay_3.GlyphTableIndexArray = 'None'
    slice_YDisplay_3.GaussianRadius = 0.5
    slice_YDisplay_3.SetScaleArray = ['POINTS', 'U235_conc']
    slice_YDisplay_3.ScaleTransferFunction = 'PiecewiseFunction'
    slice_YDisplay_3.OpacityArray = ['POINTS', 'U235_conc']
    slice_YDisplay_3.OpacityTransferFunction = 'PiecewiseFunction'
    slice_YDisplay_3.DataAxesGrid = 'GridAxesRepresentation'
    slice_YDisplay_3.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    slice_YDisplay_3.ScaleTransferFunction.Points = [-0.009985746152872028, 0.0, 0.5, 0.0, 4.3217532213886924, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    slice_YDisplay_3.OpacityTransferFunction.Points = [-0.009985746152872028, 0.0, 0.5, 0.0, 4.3217532213886924, 1.0, 0.5, 0.0]

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

    # show data from concplume
    concplumeDisplay_1 = Show(concplume, renderView4, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    concplumeDisplay_1.Representation = 'Surface'
    concplumeDisplay_1.ColorArrayName = [None, '']
    concplumeDisplay_1.SelectTCoordArray = 'None'
    concplumeDisplay_1.SelectNormalArray = 'None'
    concplumeDisplay_1.SelectTangentArray = 'None'
    concplumeDisplay_1.OSPRayScaleArray = 'U235_conc'
    concplumeDisplay_1.OSPRayScaleFunction = 'PiecewiseFunction'
    concplumeDisplay_1.SelectOrientationVectors = 'None'
    concplumeDisplay_1.ScaleFactor = 10.00000000000001
    concplumeDisplay_1.SelectScaleArray = 'None'
    concplumeDisplay_1.GlyphType = 'Arrow'
    concplumeDisplay_1.GlyphTableIndexArray = 'None'
    concplumeDisplay_1.GaussianRadius = 0.5000000000000006
    concplumeDisplay_1.SetScaleArray = ['POINTS', 'U235_conc']
    concplumeDisplay_1.ScaleTransferFunction = 'PiecewiseFunction'
    concplumeDisplay_1.OpacityArray = ['POINTS', 'U235_conc']
    concplumeDisplay_1.OpacityTransferFunction = 'PiecewiseFunction'
    concplumeDisplay_1.DataAxesGrid = 'GridAxesRepresentation'
    concplumeDisplay_1.PolarAxes = 'PolarAxesRepresentation'
    concplumeDisplay_1.ScalarOpacityUnitDistance = 3.510016309854808
    concplumeDisplay_1.OpacityArrayName = ['POINTS', 'U235_conc']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    concplumeDisplay_1.ScaleTransferFunction.Points = [1.0184191749180172e-10, 0.0, 0.5, 0.0, 2.4502793457562135, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    concplumeDisplay_1.OpacityTransferFunction.Points = [1.0184191749180172e-10, 0.0, 0.5, 0.0, 2.4502793457562135, 1.0, 0.5, 0.0]

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

    # show data from mergedoutputs
    mergedoutputsDisplay_2 = Show(mergedoutputs, renderView4, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    mergedoutputsDisplay_2.Representation = 'Wireframe'
    mergedoutputsDisplay_2.ColorArrayName = [None, '']
    mergedoutputsDisplay_2.SelectTCoordArray = 'None'
    mergedoutputsDisplay_2.SelectNormalArray = 'None'
    mergedoutputsDisplay_2.SelectTangentArray = 'None'
    mergedoutputsDisplay_2.OSPRayScaleArray = 'U235_conc'
    mergedoutputsDisplay_2.OSPRayScaleFunction = 'PiecewiseFunction'
    mergedoutputsDisplay_2.SelectOrientationVectors = 'None'
    mergedoutputsDisplay_2.ScaleFactor = 10.00000000000001
    mergedoutputsDisplay_2.SelectScaleArray = 'None'
    mergedoutputsDisplay_2.GlyphType = 'Arrow'
    mergedoutputsDisplay_2.GlyphTableIndexArray = 'None'
    mergedoutputsDisplay_2.GaussianRadius = 0.5000000000000006
    mergedoutputsDisplay_2.SetScaleArray = ['POINTS', 'U235_conc']
    mergedoutputsDisplay_2.ScaleTransferFunction = 'PiecewiseFunction'
    mergedoutputsDisplay_2.OpacityArray = ['POINTS', 'U235_conc']
    mergedoutputsDisplay_2.OpacityTransferFunction = 'PiecewiseFunction'
    mergedoutputsDisplay_2.DataAxesGrid = 'GridAxesRepresentation'
    mergedoutputsDisplay_2.PolarAxes = 'PolarAxesRepresentation'
    mergedoutputsDisplay_2.ScalarOpacityUnitDistance = 3.1509480300246997
    mergedoutputsDisplay_2.OpacityArrayName = ['POINTS', 'U235_conc']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    mergedoutputsDisplay_2.ScaleTransferFunction.Points = [-0.26518405488851754, 0.0, 0.5, 0.0, 5.134152720511603, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    mergedoutputsDisplay_2.OpacityTransferFunction.Points = [-0.26518405488851754, 0.0, 0.5, 0.0, 5.134152720511603, 1.0, 0.5, 0.0]

    # show data from extractSurface1
    extractSurface1Display = Show(extractSurface1, renderView4, 'GeometryRepresentation')

    # trace defaults for the display properties.
    extractSurface1Display.Representation = 'Wireframe'
    extractSurface1Display.ColorArrayName = [None, '']
    extractSurface1Display.SelectTCoordArray = 'None'
    extractSurface1Display.SelectNormalArray = 'None'
    extractSurface1Display.SelectTangentArray = 'None'
    extractSurface1Display.OSPRayScaleArray = 'U235_conc'
    extractSurface1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    extractSurface1Display.SelectOrientationVectors = 'None'
    extractSurface1Display.ScaleFactor = 10.00000000000001
    extractSurface1Display.SelectScaleArray = 'None'
    extractSurface1Display.GlyphType = 'Arrow'
    extractSurface1Display.GlyphTableIndexArray = 'None'
    extractSurface1Display.GaussianRadius = 0.5000000000000006
    extractSurface1Display.SetScaleArray = ['POINTS', 'U235_conc']
    extractSurface1Display.ScaleTransferFunction = 'PiecewiseFunction'
    extractSurface1Display.OpacityArray = ['POINTS', 'U235_conc']
    extractSurface1Display.OpacityTransferFunction = 'PiecewiseFunction'
    extractSurface1Display.DataAxesGrid = 'GridAxesRepresentation'
    extractSurface1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    extractSurface1Display.ScaleTransferFunction.Points = [-0.26518405488851754, 0.0, 0.5, 0.0, 5.134152720511603, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    extractSurface1Display.OpacityTransferFunction.Points = [-0.26518405488851754, 0.0, 0.5, 0.0, 5.134152720511603, 1.0, 0.5, 0.0]

    # show data from clip1
    clip1Display = Show(clip1, renderView4, 'UnstructuredGridRepresentation')

    # trace defaults for the display properties.
    clip1Display.Representation = 'Wireframe'
    clip1Display.ColorArrayName = [None, '']
    clip1Display.SelectTCoordArray = 'None'
    clip1Display.SelectNormalArray = 'None'
    clip1Display.SelectTangentArray = 'None'
    clip1Display.OSPRayScaleArray = 'U235_conc'
    clip1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    clip1Display.SelectOrientationVectors = 'None'
    clip1Display.ScaleFactor = 10.00000000000001
    clip1Display.SelectScaleArray = 'None'
    clip1Display.GlyphType = 'Arrow'
    clip1Display.GlyphTableIndexArray = 'None'
    clip1Display.GaussianRadius = 0.5000000000000006
    clip1Display.SetScaleArray = ['POINTS', 'U235_conc']
    clip1Display.ScaleTransferFunction = 'PiecewiseFunction'
    clip1Display.OpacityArray = ['POINTS', 'U235_conc']
    clip1Display.OpacityTransferFunction = 'PiecewiseFunction'
    clip1Display.DataAxesGrid = 'GridAxesRepresentation'
    clip1Display.PolarAxes = 'PolarAxesRepresentation'
    clip1Display.ScalarOpacityUnitDistance = 1.9714140946645444
    clip1Display.OpacityArrayName = ['POINTS', 'U235_conc']

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    clip1Display.ScaleTransferFunction.Points = [-0.23838177861698248, 0.0, 0.5, 0.0, 5.134152720511603, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    clip1Display.OpacityTransferFunction.Points = [-0.23838177861698248, 0.0, 0.5, 0.0, 5.134152720511603, 1.0, 0.5, 0.0]

    # show data from plane1
    plane1Display = Show(plane1, renderView4, 'GeometryRepresentation')

    # trace defaults for the display properties.
    plane1Display.Representation = 'Surface'
    plane1Display.ColorArrayName = [None, '']
    plane1Display.SelectTCoordArray = 'TextureCoordinates'
    plane1Display.SelectNormalArray = 'Normals'
    plane1Display.SelectTangentArray = 'None'
    plane1Display.OSPRayScaleArray = 'Normals'
    plane1Display.OSPRayScaleFunction = 'PiecewiseFunction'
    plane1Display.SelectOrientationVectors = 'None'
    plane1Display.ScaleFactor = 24.0
    plane1Display.SelectScaleArray = 'None'
    plane1Display.GlyphType = 'Arrow'
    plane1Display.GlyphTableIndexArray = 'None'
    plane1Display.GaussianRadius = 1.2
    plane1Display.SetScaleArray = ['POINTS', 'Normals']
    plane1Display.ScaleTransferFunction = 'PiecewiseFunction'
    plane1Display.OpacityArray = ['POINTS', 'Normals']
    plane1Display.OpacityTransferFunction = 'PiecewiseFunction'
    plane1Display.DataAxesGrid = 'GridAxesRepresentation'
    plane1Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    plane1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    plane1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # show data from plane2
    plane2Display = Show(plane2, renderView4, 'GeometryRepresentation')

    # trace defaults for the display properties.
    plane2Display.Representation = 'Surface'
    plane2Display.ColorArrayName = [None, '']
    plane2Display.SelectTCoordArray = 'TextureCoordinates'
    plane2Display.SelectNormalArray = 'Normals'
    plane2Display.SelectTangentArray = 'None'
    plane2Display.OSPRayScaleArray = 'Normals'
    plane2Display.OSPRayScaleFunction = 'PiecewiseFunction'
    plane2Display.SelectOrientationVectors = 'None'
    plane2Display.ScaleFactor = 11.0
    plane2Display.SelectScaleArray = 'None'
    plane2Display.GlyphType = 'Arrow'
    plane2Display.GlyphTableIndexArray = 'None'
    plane2Display.GaussianRadius = 0.55
    plane2Display.SetScaleArray = ['POINTS', 'Normals']
    plane2Display.ScaleTransferFunction = 'PiecewiseFunction'
    plane2Display.OpacityArray = ['POINTS', 'Normals']
    plane2Display.OpacityTransferFunction = 'PiecewiseFunction'
    plane2Display.DataAxesGrid = 'GridAxesRepresentation'
    plane2Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    plane2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    plane2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

    # show data from slice3
    slice3Display = Show(slice3, renderView4, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice3Display.Representation = 'Surface'
    slice3Display.ColorArrayName = ['POINTS', 'U235_conc']
    slice3Display.LookupTable = u235_concLUT
    slice3Display.SelectTCoordArray = 'None'
    slice3Display.SelectNormalArray = 'None'
    slice3Display.SelectTangentArray = 'None'
    slice3Display.OSPRayScaleArray = 'U235_conc'
    slice3Display.OSPRayScaleFunction = 'PiecewiseFunction'
    slice3Display.SelectOrientationVectors = 'None'
    slice3Display.ScaleFactor = 10.000000000000002
    slice3Display.SelectScaleArray = 'None'
    slice3Display.GlyphType = 'Arrow'
    slice3Display.GlyphTableIndexArray = 'None'
    slice3Display.GaussianRadius = 0.5000000000000001
    slice3Display.SetScaleArray = ['POINTS', 'U235_conc']
    slice3Display.ScaleTransferFunction = 'PiecewiseFunction'
    slice3Display.OpacityArray = ['POINTS', 'U235_conc']
    slice3Display.OpacityTransferFunction = 'PiecewiseFunction'
    slice3Display.DataAxesGrid = 'GridAxesRepresentation'
    slice3Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    slice3Display.ScaleTransferFunction.Points = [-0.033120488635322756, 0.0, 0.5, 0.0, 0.28330347210762097, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    slice3Display.OpacityTransferFunction.Points = [-0.033120488635322756, 0.0, 0.5, 0.0, 0.28330347210762097, 1.0, 0.5, 0.0]

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

    # show data from slice4
    slice4Display = Show(slice4, renderView4, 'GeometryRepresentation')

    # trace defaults for the display properties.
    slice4Display.Representation = 'Surface'
    slice4Display.ColorArrayName = [None, '']
    slice4Display.SelectTCoordArray = 'None'
    slice4Display.SelectNormalArray = 'None'
    slice4Display.SelectTangentArray = 'None'
    slice4Display.OSPRayScaleArray = 'U235_conc'
    slice4Display.OSPRayScaleFunction = 'PiecewiseFunction'
    slice4Display.SelectOrientationVectors = 'None'
    slice4Display.ScaleFactor = 10.000000000000002
    slice4Display.SelectScaleArray = 'None'
    slice4Display.GlyphType = 'Arrow'
    slice4Display.GlyphTableIndexArray = 'None'
    slice4Display.GaussianRadius = 0.5000000000000001
    slice4Display.SetScaleArray = ['POINTS', 'U235_conc']
    slice4Display.ScaleTransferFunction = 'PiecewiseFunction'
    slice4Display.OpacityArray = ['POINTS', 'U235_conc']
    slice4Display.OpacityTransferFunction = 'PiecewiseFunction'
    slice4Display.DataAxesGrid = 'GridAxesRepresentation'
    slice4Display.PolarAxes = 'PolarAxesRepresentation'

    # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
    slice4Display.ScaleTransferFunction.Points = [-0.018342102480340454, 0.0, 0.5, 0.0, 0.1341648618140519, 1.0, 0.5, 0.0]

    # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
    slice4Display.OpacityTransferFunction.Points = [-0.018342102480340454, 0.0, 0.5, 0.0, 0.1341648618140519, 1.0, 0.5, 0.0]

    # setup the color legend parameters for each legend in this view

    # get color legend/bar for u235_concLUT in view renderView4
    u235_concLUTColorBar_2 = GetScalarBar(u235_concLUT, renderView4)
    u235_concLUTColorBar_2.Orientation = 'Horizontal'
    u235_concLUTColorBar_2.WindowLocation = 'Any Location'
    u235_concLUTColorBar_2.Position = [0.11326500732064429, 0.8591767781892214]
    u235_concLUTColorBar_2.Title = 'U235_conc'
    u235_concLUTColorBar_2.ComponentTitle = ''
    u235_concLUTColorBar_2.ScalarBarLength = 0.4368814055636897

    # set color bar visibility
    u235_concLUTColorBar_2.Visibility = 1

    # show color legend
    macroFrDisplay_1.SetScalarBarVisibility(renderView4, True)

    # hide data in view
    Hide(fractures, renderView4)

    # show color legend
    slice_YDisplay_3.SetScalarBarVisibility(renderView4, True)

    # hide data in view
    Hide(slice_Y, renderView4)

    # show color legend
    slice_ZDisplay.SetScalarBarVisibility(renderView4, True)

    # hide data in view
    Hide(concplume, renderView4)

    # show color legend
    microFrDisplay_1.SetScalarBarVisibility(renderView4, True)

    # hide data in view
    Hide(mergedoutputs, renderView4)

    # hide data in view
    Hide(extractSurface1, renderView4)

    # hide data in view
    Hide(clip1, renderView4)

    # hide data in view
    Hide(plane1, renderView4)

    # hide data in view
    Hide(plane2, renderView4)

    # show color legend
    slice3Display.SetScalarBarVisibility(renderView4, True)

    # ----------------------------------------------------------------
    # setup color maps and opacity mapes used in the visualization
    # note: the Get..() functions create a new object, if needed
    # ----------------------------------------------------------------

    # get opacity transfer function/opacity map for 'piezo_head_p0'
    piezo_head_p0PWF = GetOpacityTransferFunction('piezo_head_p0')
    piezo_head_p0PWF.Points = [656.4727751933933, 0.0, 0.5, 0.0, 660.0745594750949, 1.0, 0.5, 0.0]
    piezo_head_p0PWF.ScalarRangeInitialized = 1

    # ----------------------------------------------------------------
    # restore active source
    SetActiveSource(slice4)
    # ----------------------------------------------------------------

    SaveExtracts(ExtractsOutputDirectory='extracts')


# if __name__ == '__main__':
#     # generate extracts
#     SaveExtracts(ExtractsOutputDirectory='extracts')
