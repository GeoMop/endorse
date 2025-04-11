import bgem.gmsh.gmsh
import matplotlib.pyplot as plt
from endorse import common
import numpy as np
import pandas as pd
from math import sin, cos, pi
#from bgem import gmsh, geometry
import yaml
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as ticker



class Boreholes:
    def __init__(self, yaml_config_file):
        self.config = common.config.load_config(yaml_config_file)


    @property
    def l5_width(self):
        return self.config.geometry.l5_width

    @property
    def l5_azimuth(self):
        return self.config.geometry.l5_azimuth

    @property
    def y_offset(self):
        return self.config.geometry.y_offset

    def bh_name(self, bh_index):
        return self.config.boreholes[bh_index].name

    def bh_id(self, bh_index):
        return self.config.boreholes[bh_index].id

    def bh_index_from_name(self, bh_name):
        idx = next(i for i,bh in enumerate(self.config.boreholes) if bh.name == bh_name)
        return idx

    def bh_start(self, bh_index):
        bh = self.config.boreholes[bh_index]
        # We detect orientation from borehole id (starts either 'L' or 'P')
        bh_orientation = 1 if self.bh_id(bh_index)[0] == 'P' else -1
        x = bh_orientation * self.l5_width / 2
        y = (bh.stationing - self.y_offset)
        z = bh.starting_height
        return np.array([x, y, z])

    def bh_direction(self, bh_index):
        bh = self.config.boreholes[bh_index]
        x = cos((self.l5_azimuth + 90 - bh.azimuth) * pi / 180)
        y = sin((self.l5_azimuth + 90 - bh.azimuth) * pi / 180)
        z = sin(bh.inclination * pi / 180)
        return np.array([x, y, z])

    @property
    def n_boreholes(self):
        return len(self.config.boreholes)

    def n_chambers(self, bh_index):
        return len(self.config.boreholes[bh_index].packer_centers)

    def chamber_name(self, bh_index, chamber_index):
        return self.bh_name(bh_index) + f"_ch_{chamber_index}"

    def chamber_start(self, bh_index, chamber_index):
        bh = self.config.boreholes[bh_index]
        chamber_start_distance = bh.packer_centers[chamber_index] + bh.packer_width / 2
        return self.bh_start(bh_index) + chamber_start_distance*self.bh_direction(bh_index)

    def chamber_center(self, bh_index, chamber_index):
        bh = self.config.boreholes[bh_index]
        if chamber_index+1 < len(bh.packer_centers):
            chamber_center_distance = (bh.packer_centers[chamber_index] + bh.packer_centers[chamber_index+1]) / 2
        else:
            chamber_center_distance = (bh.packer_centers[chamber_index] + bh.packer_width/2 + bh.length) / 2
        return self.bh_start(bh_index) + chamber_center_distance * self.bh_direction(bh_index)

    def chamber_end(self, bh_index, chamber_index):
        bh = self.config.boreholes[bh_index]
        if chamber_index+1 < len(bh.packer_centers):
            chamber_end_distance = bh.packer_centers[chamber_index+1] - bh.packer_width / 2
        else:
            chamber_end_distance = bh.length
        return self.bh_start(bh_index) + chamber_end_distance * self.bh_direction(bh_index)

    def sensor_position(self, bh_index, chamber_index):
        return self.config.boreholes[bh_index].sensor_positions[chamber_index]

    def make_gmsh_lines(self, factory: bgem.gmsh.gmsh.GeometryOCC):
        lines = []
        for i in range(self.n_boreholes):
            n_chambers = self.n_chambers(i)
            for ch in range(n_chambers):
                p1 = factory.point(self.chamber_start(i,ch))
                p2 = factory.point(self.chamber_end(i, ch))
                lines.append( factory.line(p1, p2) )
        return lines

    def make_gmsh_rectangles(self, factory: bgem.gmsh.gmsh.GeometryOCC):
        rects = []
        for i in range(self.n_boreholes):
            n_chambers = self.n_chambers(i)
            for ch in range(n_chambers):
                p1 = factory.point(self.chamber_start(i,ch))
                p2 = factory.point(self.chamber_end(i, ch))
                line = factory.line(p1, p2)
                rects.append( line.extrude([0,0,0.1])[2] )
                # factory.model.remove(line.dim_tags)
        print(rects)
        group = factory.group(*rects)
        factory.make_mesh([group])
        factory.write_mesh('rects.msh')
        return rects

    def make_observe_points(self, output_file: str, n_pts_per_chamber=3):
        point_list = []
        for i in range(self.n_boreholes):
            n_chambers = self.n_chambers(i)
            for ch in range(n_chambers):
                p1 = self.chamber_start(i,ch)
                p2 = self.chamber_end(i, ch)
                pts = np.linspace(p1, p2, n_pts_per_chamber)
                ch_name = self.chamber_name(i, ch)
                list = [{"name": ch_name + f"_pt_{ipt}", "point": pt.tolist()} for ipt,pt in enumerate(pts)]
                point_list.extend(list)

        with open(output_file, "w") as file:
            yaml.dump(point_list, file, default_flow_style=True)



class ObservePointData:

    def __init__(self, bhs: Boreholes, input_yaml_file: str):
        self._bhs = bhs
        self._data = common.config.load_config(input_yaml_file)
        self._pts = [pt.observe_point for pt in self._data.points]
        self._pt_names = [pt.name for pt in self._data.points]

        self._chamber_names = []
        for bi in range(self._bhs.n_boreholes):
            for ci in range(self._bhs.n_chambers(bi)):
                self._chamber_names.append(self._bhs.chamber_name(bi,ci))

        self._pt_to_chamber = []
        for i_pt,pt_name in enumerate(self._pt_names):
            ch_name = pt_name[0:pt_name.rfind("_pt_")]
            self._pt_to_chamber.append(self._chamber_names.index(ch_name))
        self._pt_to_chamber = np.array(self._pt_to_chamber)

        self._pressures = {}
        self._pressure_bounds = np.zeros((len(self._chamber_names),2))
        self._times = []
        for time_data in self._data.data:
            t = float(time_data.time)
            self._times.append(t)
            self._pressures[t] = np.array(time_data.pressure_p0)
            for i,p in enumerate(self._pressures[t]):
                ch = self._pt_to_chamber[i]
                self._pressure_bounds[ch,0] = min(self._pressure_bounds[ch,0], p)
                self._pressure_bounds[ch][1] = max(self._pressure_bounds[ch,1], p)

    @property
    def chamber_names(self):
        return self._chamber_names

    def chamber_pressures(self, chamber_index, time):
        return self._pressures[time][self._pt_to_chamber==chamber_index]

    def animate(self):
        def update(time_index):
            for ci in range(len(self._chamber_names)):
                pressures = self.chamber_pressures(ci, self._times[time_index])
                lines[ci].set_ydata(pressures)
            time_text.set_text(f"Time: {self._times[time_index]}")
            print(self._times[time_index])
            l = lines + [time_text]
            return l

        fig, ax = plt.subplots(3,8, figsize=(24,6), subplot_kw={"projection":"3d"})
        ax = ax.transpose().flatten()
        lines = []
        for ci,cname in enumerate(self._chamber_names):
            pressures = self.chamber_pressures(ci, 0)
            xx = np.linspace(0,1,len(pressures))
            ax[ci].set_title(cname)
            ax[ci].set_zlim(self._pressure_bounds[ci])
            ax[ci].set_ylim(100,130)
            ax[ci].set_xlim(0,len(xx)-1)
            for t in reversed(self._times):
                pressures = self.chamber_pressures(ci, t)
                line, = ax[ci].plot(xx, t*np.ones(len(xx)), pressures)
                lines.append(line)
        time_text = fig.suptitle(f"Time: 0", fontsize=14)

        # ani = animation.FuncAnimation(fig, update, frames=len(self._times), interval=1, blit=False)
        plt.tight_layout()
        plt.show()

    def plot_chamber_pressure_averages(self, output_file=None, print_pressures=False):
        fig, ax = plt.subplots(3, 8, figsize=(60, 15))
        ax = ax.transpose().flatten()
        for ci, cname in enumerate(self._chamber_names):
            xx = self._times
            pressures = np.zeros(len(xx))
            for i,t in enumerate(self._times):
                pressures[i] = np.mean(self.chamber_pressures(ci, t))
            ax[ci].set_title(cname)
            # ax[ci].set_ylim(self._pressure_bounds[ci])
            ax[ci].set_xlim(100, 130)
            #ax[ci].yaxis.set_major_locator(ticker.MultipleLocator(50))
            #ax[ci].yaxis.set_minor_locator(ticker.MultipleLocator(10))
            #ax[ci].minorticks_on()
            ax[ci].plot(xx, pressures)
            if print_pressures:
                print(f"Chamber {cname} initial pressure at t=100: {pressures[1]}")

        plt.tight_layout()
        if output_file is None:
            plt.show()
        else:
            plt.savefig(output_file)

        return fig, ax


def plot_chamber_pressures(boreholes_fname,
                           pressure_fname,
                           output_fname,
                           pressure_init_fname=None,
                           output_init_fname=None,
                           pressure_ref_csv_fname=None,
                           output_compared_fname=None
                          ):
    bhs = Boreholes(boreholes_fname)
    obs = ObservePointData(bhs, pressure_fname)
    fig, ax = obs.plot_chamber_pressure_averages(output_fname)

    # shift time scale and add labels
    for axis in ax:
        for line in axis.get_lines():
            line.set_xdata(line.get_xdata() - 100)
            line.set_label('model pressure [m]')
            axis.set_xlim(0,60)
            axis.legend()
    
    if pressure_init_fname is not None:
        obs_i = ObservePointData(bhs, pressure_init_fname)
        fig_i, ax_i = obs_i.plot_chamber_pressure_averages(output_init_fname)

        # plot both lines to one plot
        for axis, axis_i in zip(ax, ax_i):
            for line in axis_i.get_lines():
                axis.plot(line.get_xdata() - 100, line.get_ydata(), label='model with enforced initial pressure [m]')
                axis.legend()

    # read and plot reference pressures from measurements
    if pressure_ref_csv_fname is not None:
        df = pd.read_csv(pressure_ref_csv_fname, delimiter=';')
        for axis,cname in zip(ax,obs.chamber_names):
            bhname, _, cidx = cname.rpartition('_ch_')
            bh_index = bhs.bh_index_from_name(bhname)
            cidx = int(cidx)
            pos = bhs.sensor_position(bh_index, cidx)
            axis.set_title(f"{bhname} chamber {cidx+1} position {pos} m")
            filtered_df = df[(df['Borehole'] == bhname) & (df['Chamber'] == cidx)]
            sim_time = filtered_df['sim_time']
            pressure = filtered_df['tlak'].to_numpy() / 10 # from kPa to m
            if pressure.size > 0:
                axis.plot(sim_time, pressure, label='measured pressure [m]')
                axis.legend()

    if output_compared_fname is None:
        fig.show()
    else:
        fig.savefig(output_compared_fname)

if __name__ == "__main__":
    # plot comparison of model and measured pressure in chambers
    borehole_fname = 'boreholes.yaml'
    pressure_fname = 'flow_observe_refined.yaml'
    output_fname = 'chamber_pressure_averages_refined.pdf'
    pressure_init_fname = 'flow_observe_refined_init_p.yaml'
    output_init_fname = 'chamber_pressure_averages_refined_init_p.pdf'
    pressure_ref_fname = 'output_tlaky.csv'
    output_compared_fname = 'chamber_pressures_refined_compared.pdf'
    plot_chamber_pressures(borehole_fname,
                           pressure_fname,
                           output_fname,
                           pressure_ref_csv_fname=pressure_ref_fname,
                           output_compared_fname=output_compared_fname)
