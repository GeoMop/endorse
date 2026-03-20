from paraview.simple import *
from paraview.servermanager import Fetch
import vtk
import pathlib
import os

script_dir = pathlib.Path(__file__).parent
datafile_name = "boreholes.vtm"

os.chdir(script_dir)
view = GetActiveViewOrCreate('RenderView')

# for proxy in GetSources().values():
#     if GetDisplayProperties(proxy, view=view).Visibility == 1:
#         Show(proxy, view)

# ==== Načtení VTK souboru ====
reader = XMLMultiBlockDataReader(FileName=[datafile_name])
RenameProxy(reader, group=reader.GetXMLGroup(), newName="Boreholes")
Show(reader, view)

vtk_data = Fetch(reader)

# === Přístup k FieldData ===
field_data = vtk_data.GetFieldData()

labels_array = field_data.GetAbstractArray('Labels')
nlabels = labels_array.GetNumberOfTuples()

labels = [labels_array.GetValue(i) for i in range(nlabels)]

# připrav lookup table pro bh_index
lut = GetColorTransferFunction("bh_index")
lut.AnnotationsInitialized = 1
lut.InterpretValuesAsCategories = 1
annotations = []
for i,l in enumerate(labels):
    annotations.extend([str(i),l])
lut.Annotations = annotations
# lut.IndexedColors - lze nastavit RGB hodnoty barev pro jednotlive kategorie=vrty
# lut.IndexedColors[21] = 0
# lut.IndexedColors[22] = 0
# lut.IndexedColors[23] = 0

# Nezobrazovat text "bh_index" v legende.
bar = GetScalarBar(lut)
bar.Title = ""
bar.ComponentTitle = "Boreholes"
bar.Visibility = 1


# Vytvoření textových popisků
coords_array = field_data.GetAbstractArray("LabelCoords")

# --- vytvořit vtkPolyData s body ---
points = vtk.vtkPoints()
points.SetNumberOfPoints(nlabels)

labels = vtk.vtkStringArray()
labels.SetName("Labels")
labels.SetNumberOfTuples(nlabels)

for i in range(nlabels):
    x, y, z = coords_array.GetTuple(i)
    points.SetPoint(i, x, y, z)
    labels.SetValue(i, labels_array.GetValue(i))

poly = vtk.vtkPolyData()
poly.SetPoints(points)
poly.GetPointData().AddArray(labels)


# --- zabalit do ParaView source ---
label_source = TrivialProducer()
label_source.GetClientSideObject().SetOutput(poly)
label_source.UpdatePipeline()

label_source.Rename("Borehole Labels")

display = Show(label_source)
display.Representation = "Labels"
display.PointFieldDataArrayName = "Labels"
display.PointLabelVisibility = 1
display.PointLabelJustification = "Center"

# aplikuj barvení a zobraz legendu
# Show(reader, view)
Render()