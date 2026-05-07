import numpy as np
import pandas as pd
import pyvista as pv
from math import pi
import boreholes

def export_fractures(zone_name = "L5"):
    # Visualize fractures along L5 gallery as circular disks. Save as VTK.

    origin_local = [ -622600, -1127800, 0 ]
    origin_sim = [-61.66, -22.71, 18]
    l5_azimuth = 110
    arrow_scale = 2
    disk_radius = 1
    disk_resolution = 50


    # načtení ODS souboru
    df = pd.read_excel("../input_data/EP02_PVP2_ZZ_Tabulka_vysledky_dokumentace_celeb_a_chodeb.ods",
                       sheet_name="Dokumentace_stěn",
                       engine="odf",
                       header=1)

    # výběr podle podmínek
    filtered = df[
        (df["Chodba, větrací chodba, zkušební komora"] == zone_name)
      # & (df["Výplň struktury (cc-kalcit, chl-chlorit, sulf-sulfidy, py-pyrit, q-křemen)"].isna())
    ]

    # vytvoření numpy matice souřadnic
    points = filtered[["X [m] S-JTSK", "Y [m] S-JTSK", "Z [m] Bpv"]].to_numpy()
    angles = filtered[["Směr sklonu struktury [°] v S-JTSK", "Sklon struktury [°]"]].to_numpy()

    # posun do lokálních souřadnic
    points -= np.array(origin_local)

    # posun do souřadnic pro simulaci
    points -= np.array(origin_sim)

    # úhel v radiánech
    theta = np.deg2rad(l5_azimuth)

    # rotační matice kolem Z
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    # otočené body
    points @= Rz.T   # násobíme zprava transponovanou maticí

    # výběr bodů s Y=souřadnicí -25 až 25
    mask = (points[:, 1] >= -25) & (points[:, 1] <= 25)
    points = points[mask]
    angles = angles[mask]

    normals = np.zeros((points.shape[0],3))
    normals[:,0] = np.cos((l5_azimuth + 90 - angles[:,0]) * pi / 180)
    normals[:,1] = np.sin((l5_azimuth + 90 - angles[:,0]) * pi / 180)
    normals[:,2] = np.sin(angles[:,1] * pi / 180)

    # vytvoření VTK objektu s body
    cloud = pv.PolyData(points)

    # vytvoření všech šipek
    arrows = pv.MultiBlock()
    for p,direction in zip(points,normals):
        arrow = pv.Arrow(start=p, direction=direction, scale=arrow_scale)
        arrows.append(arrow)

    # vytvoření všech disků
    disks = pv.MultiBlock()
    for p,normal in zip(points,normals):
        disk = pv.Disc(center=p, normal=normal, inner=0, outer=disk_radius, r_res=disk_resolution, c_res=disk_resolution)
        disks.append(disk)

    # spojení bodů a šipek
    scene = disks.combine()

    # uložení do souboru
    scene.save(f"fractures-{zone_name}.vtk")



def visualize_fractures(output_file):
    # Visualize fractures along boreholes as circular disks. Save as VTK.

    bhs = boreholes.Boreholes()

    # Number of points to discretize each circle boundary
    n_circle_points = 40
    # Radius of all fractures
    radius = 2

    all_meshes = []
    for bi in range(bhs.n_boreholes):
        bh_start = bhs.bh_start(bi)
        bh_normal = bhs.bh_direction(bi)
        for fi in range(bhs.n_fractures(bi)):
            f = bhs.fracture(bi, fi)
            f_center = bh_start + bh_normal * f.position
            f_normal = bhs.fr_normal(bi,fi)
            disc = pv.Disc(center=f_center, inner=0.0, outer=radius, normal=f_normal, c_res=n_circle_points)

            # Assign scalars as per-disc (cell data)
            n_cells = disc.n_cells
            disc.cell_data["width"] = np.full(n_cells, f.width)
            disc.cell_data["flag"] = np.full(n_cells, f.flag)

            all_meshes.append(disc)

    # Combine all discs into a single mesh
    combined = all_meshes[0]
    for m in all_meshes[1:]:
        combined = combined.merge(m)

    # Save to VTU file
    combined.save(output_file)




if __name__ == "__main__":
    export_fractures("L5")
    export_fractures("ZK5-1S")
    export_fractures("ZK5-1J")
    visualize_fractures("fractures.vtk")