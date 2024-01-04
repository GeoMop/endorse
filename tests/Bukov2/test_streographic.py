import pytest
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.ticker import MaxNLocator, FuncFormatter
import cartopy.feature as cfeature
import numpy as np


#@pytest.mark.skip
def test_full_stereo():
    def sphere_circle(lon, lat, r_phi, n_points):
        lat0, lon0, phi = np.radians([lat, lon, r_phi])  # Convert to radians
        circle_points = []

        theta = np.linspace(0, 2 * np.pi, n_points)
        lat = np.arcsin(np.sin(lat0) * np.cos(phi) + np.cos(lat0) * np.sin(phi) * np.cos(theta))
        lon = lon0 + np.arctan2(np.sin(theta) * np.sin(phi) * np.cos(lat0),
                                np.cos(phi) - np.sin(lat0) * np.sin(lat))

        # Convert back to degrees
        lat, lon = np.degrees([lat, lon])

        lon = lon % 360
        return list(zip(lon, lat))
    # Example usage
    p_dict = dict(
        bh=[(81.2, 9.45, "BH0"), (312.6, 16.7, "BH1")],
        F1=[(265, 45)],
        allowed=sphere_circle(265, 45, 60, 72),
        critical=sphere_circle(265, 45, 80, 72)
    )

    fragile_stereo_projection(p_dict)



def fragile_stereo_projection(p_dict):
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    import matplotlib.path as mpath
    import numpy as np

    # Value for r_extent is obtained by trial and error
    # get it with `ax.get_ylim()` after running this code
    #r_extent = 4651194.319
    #r_extent *= 1.005  # increase a bit for better result
    r_extent = 1.3 * 1e7


    # Projection settings
    lonlat_proj = ccrs.PlateCarree()
    use_proj = ccrs.SouthPolarStereo(central_longitude=0)
    fig = plt.figure(figsize=[7, 7])
    ax = plt.subplot(1, 1, 1, projection=use_proj)
    ax.set_extent([-180, 180, 0, 90], lonlat_proj)

    # ax.stock_img() # add bluemarble image
    #ax.coastlines(lw=0.5, color="black", zorder=20)  # add coastlines

    # draw graticule (meridian and parallel lines)
    gls = ax.gridlines(draw_labels=True, crs=lonlat_proj, lw=1, color="gray",
                       y_inline=True, xlocs=range(-180, 180, 10), ylocs=range(-90, 1, 10))

    def longitude_formatter(lon, pos):
        return f'{lon + 360 if lon < 0 else lon}°'

    gls.xformatter = FuncFormatter(longitude_formatter)

    # set the plot limits
    ax.set_xlim(-r_extent, r_extent)
    ax.set_ylim(-r_extent, r_extent)

    set1_cmap = plt.get_cmap('Set1')
    text_offset = 0.5
    add_label = lambda l: "" if l in ax.get_legend_handles_labels()[1] else l

    def fix_point(tpl):
        if len(tpl) < 3:
            tpl = (*tpl, "")
        azim, inkl, txt = tpl
        if inkl > 0:
            azim, inkl = (azim + 180) % 360, -inkl
        return inkl, azim, txt


    for i_set, (label, data) in enumerate(p_dict.items()):
        # Plot each point in the first set
        data = [fix_point(tpl) for tpl in data]
        color = set1_cmap(i_set)
        #lat, lon, notes = zip(*data)
        for lat,lon, txt in data:
            ax.scatter(lon, lat, s=10, color=color, transform=lonlat_proj, label=add_label(label))
            if txt != "":
                ax.text(lon + text_offset, lat + text_offset, txt, transform=ccrs.PlateCarree(),
                        ha='left', va='bottom', color='blue')


    # Prep circular boundary
    circle_path = mpath.Path.unit_circle()
    circle_path = mpath.Path(circle_path.vertices.copy() * r_extent,
                             circle_path.codes.copy())
    #path=mpath.Path([(-180,0), (180,0)])
    # set circular boundary
    # this method will not interfere with the gridlines' labels
    #ax.set_boundary(path, transform=lonlat_proj)
    ax.set_boundary(circle_path)
    ax.set_frame_on(False)  # hide the boundary frame

    plt.draw()  # Enable the use of `gl._labels`


    # Add a legend
    ax.legend(loc='upper right')
    plt.show()