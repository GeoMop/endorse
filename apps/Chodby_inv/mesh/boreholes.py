import bgem.gmsh.gmsh
from endorse import common
import numpy as np
from math import sin, cos, pi
from bgem import gmsh, geometry


class Boreholes:
    def __init__(self, yaml_config_file):
        self.config = common.config.load_config(yaml_config_file)
        self.bh_list = []


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

    def make_gmsh_lines(self, factory: bgem.gmsh.gmsh.GeometryOCC):
        lines = []
        for i in range(self.n_boreholes):
            n_chambers = self.n_chambers(i)
            for ch in range(n_chambers):
                p1 = factory.point(self.chamber_start(i,ch))
                p2 = factory.point(self.chamber_end(i, ch))
                lines.append( factory.line(p1, p2) )
        return lines


if __name__ == "__main__":
    bhs = Boreholes('boreholes.yaml')
    print(np.array([[bhs.chamber_center(bi,ci) for ci in [0,1,2]] for bi in range(bhs.n_boreholes)]))