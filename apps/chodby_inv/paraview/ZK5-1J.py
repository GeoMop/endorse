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
renderView1.ViewSize = [1422, 822]
renderView1.InteractionMode = '2D'
renderView1.AxesGrid = 'GridAxes3DActor'
renderView1.CenterOfRotation = [-68.34375, -26.9375, 20.501097679138184]
renderView1.StereoType = 'Crystal Eyes'
renderView1.CameraPosition = [-63.32483111605353, -32.33926513250136, 53.74731369613767]
renderView1.CameraFocalPoint = [-63.32483111605353, -32.33926513250136, 20.501097679138184]
renderView1.CameraFocalDisk = 1.0
renderView1.CameraParallelScale = 8.604753882791925
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
layout1.SetSize(1422, 822)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView1)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'PNG Series Reader'
pNGSeriesReader1 = PNGSeriesReader(registrationName='PNGSeriesReader1', FileNames=['PVP Bukov II k 08-2024 bez stufu, srafa.png'])

# create a new 'Transform'
transform2 = Transform(registrationName='Transform2', Input=pNGSeriesReader1)
transform2.Transform = 'Transform'

# init the 'Transform' selected for 'Transform'
transform2.Transform.Translate = [-150.4, -216.7, 0.0]
transform2.Transform.Scale = [0.09532888465204957, 0.09532888465204957, 1.0]

# create a new 'Wavefront OBJ Reader'
pVP_Bukov_II_Model_a_data_2025obj = WavefrontOBJReader(registrationName='PVP_Bukov_II_Model_a_data_2025.obj', FileName='/home/jb/workspace/endorse/apps/chodby_inv/paraview/PVP_Bukov_II_Model_a_data_2025.obj')

# create a new 'Transform'
transform1 = Transform(registrationName='Transform1', Input=pVP_Bukov_II_Model_a_data_2025obj)
transform1.Transform = 'Transform'
transform1.TransformAllInputVectors = 0

# init the 'Transform' selected for 'Transform'
transform1.Transform.Translate = [622600.0, 1127800.0, 0.0]

# ----------------------------------------------------------------
# setup the visualization in view 'renderView1'
# ----------------------------------------------------------------

# show data from transform1
transform1Display = Show(transform1, renderView1, 'GeometryRepresentation')

# a texture
imageTexture1 = CreateTexture('/home/jb/workspace/endorse/apps/chodby_inv/paraview/ZK5-1J_PTG_SJTSK.png')

# trace defaults for the display properties.
transform1Display.Representation = 'Surface'
transform1Display.ColorArrayName = ['POINTS', '']
transform1Display.SelectTCoordArray = 'ZK5-1J_PTG_SJTSK'
transform1Display.SelectNormalArray = 'None'
transform1Display.SelectTangentArray = 'None'
transform1Display.Texture = imageTexture1
transform1Display.OSPRayScaleArray = 'ZK5-1J_PTG_SJTSK'
transform1Display.OSPRayScaleFunction = 'PiecewiseFunction'
transform1Display.SelectOrientationVectors = 'None'
transform1Display.ScaleFactor = 1.3125
transform1Display.SelectScaleArray = 'None'
transform1Display.GlyphType = 'Arrow'
transform1Display.GlyphTableIndexArray = 'None'
transform1Display.GaussianRadius = 0.065625
transform1Display.SetScaleArray = ['POINTS', 'ZK5-1J_PTG_SJTSK']
transform1Display.ScaleTransferFunction = 'PiecewiseFunction'
transform1Display.OpacityArray = ['POINTS', 'ZK5-1J_PTG_SJTSK']
transform1Display.OpacityTransferFunction = 'PiecewiseFunction'
transform1Display.DataAxesGrid = 'GridAxesRepresentation'
transform1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
transform1Display.ScaleTransferFunction.Points = [7.000000186963007e-05, 0.0, 0.5, 0.0, 0.9995070099830627, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
transform1Display.OpacityTransferFunction.Points = [7.000000186963007e-05, 0.0, 0.5, 0.0, 0.9995070099830627, 1.0, 0.5, 0.0]

# show data from transform2
transform2Display = Show(transform2, renderView1, 'StructuredGridRepresentation')

# get color transfer function/color map for 'PNGImage'
pNGImageLUT = GetColorTransferFunction('PNGImage')
pNGImageLUT.RGBPoints = [255.0, 0.231373, 0.298039, 0.752941, 382.5, 0.865003, 0.865003, 0.865003, 510.0, 0.705882, 0.0156863, 0.14902]
pNGImageLUT.ScalarRangeInitialized = 1.0

# get opacity transfer function/opacity map for 'PNGImage'
pNGImagePWF = GetOpacityTransferFunction('PNGImage')
pNGImagePWF.Points = [255.0, 0.0, 0.5, 0.0, 510.0, 1.0, 0.5, 0.0]
pNGImagePWF.ScalarRangeInitialized = 1

# trace defaults for the display properties.
transform2Display.Representation = 'Surface'
transform2Display.ColorArrayName = ['POINTS', 'PNGImage']
transform2Display.LookupTable = pNGImageLUT
transform2Display.MapScalars = 0
transform2Display.Opacity = 0.37
transform2Display.SelectTCoordArray = 'None'
transform2Display.SelectNormalArray = 'None'
transform2Display.SelectTangentArray = 'None'
transform2Display.OSPRayScaleArray = 'PNGImage'
transform2Display.OSPRayScaleFunction = 'PiecewiseFunction'
transform2Display.SelectOrientationVectors = 'None'
transform2Display.ScaleFactor = 33.42230695900858
transform2Display.SelectScaleArray = 'None'
transform2Display.GlyphType = 'Arrow'
transform2Display.GlyphTableIndexArray = 'PNGImage'
transform2Display.GaussianRadius = 1.671115347950429
transform2Display.SetScaleArray = ['POINTS', 'PNGImage']
transform2Display.ScaleTransferFunction = 'PiecewiseFunction'
transform2Display.OpacityArray = ['POINTS', 'PNGImage']
transform2Display.OpacityTransferFunction = 'PiecewiseFunction'
transform2Display.DataAxesGrid = 'GridAxesRepresentation'
transform2Display.PolarAxes = 'PolarAxesRepresentation'
transform2Display.ScalarOpacityFunction = pNGImagePWF
transform2Display.ScalarOpacityUnitDistance = 1.9908831827774869

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
transform2Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 255.0, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
transform2Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 255.0, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for pNGImageLUT in view renderView1
pNGImageLUTColorBar = GetScalarBar(pNGImageLUT, renderView1)
pNGImageLUTColorBar.Title = 'PNGImage'
pNGImageLUTColorBar.ComponentTitle = 'Magnitude'

# set color bar visibility
pNGImageLUTColorBar.Visibility = 1

# show color legend
transform2Display.SetScalarBarVisibility(renderView1, True)

# ----------------------------------------------------------------
# setup color maps and opacity mapes used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# restore active source
SetActiveSource(transform2)
# ----------------------------------------------------------------


if __name__ == '__main__':
    # generate extracts
    SaveExtracts(ExtractsOutputDirectory='extracts')