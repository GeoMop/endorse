from typing import *
import attrs

import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline

from .common import File, report, memoize, year


@attrs.define
class Extractor:
    attr_name: str
    z_loc: Tuple[float, float]
    _extract_data:Callable[[Any, str], np.array]

    @staticmethod
    def _extract_point_data(dataset, attr):
        return dataset.point_data[attr]

    @staticmethod
    def _extract_cell_data(dataset, attr):
        return dataset.cell_data[attr]

    @classmethod
    def from_point_data(cls, attr_name, z_loc):
        return cls(attr_name, z_loc, cls._extract_point_data)

    @classmethod
    def from_cell_data(cls, attr_name, z_loc):
        return cls(attr_name, z_loc, cls._extract_cell_data)

    def __call__(self, dataset):
        plane1 = dataset.slice(normal=[0, 0, 1], origin=[0, 0, self.z_loc[0]])
        plane2 = dataset.slice(normal=[0, 0, 1], origin=[0, 0, self.z_loc[1]])
        surface = plane1[0].merge(plane2[0])
        return self._extract_data(surface, self.attr_name)

@attrs.define
class Indicator:
    indicator_label:str
    indicator_label_short: str
    red_op: str
    q_exp:float

    def reduction(self, x):
        return getattr(self, self.red_op)(x, self.q_exp)

    @staticmethod
    def _quantile(x, q):
        return np.quantile(x, q)

    @staticmethod
    def _max(x, q):
        return np.max(x)

    @classmethod
    def quantile(cls, q:float):
        """
        Can not use lambdas here since memoize can not pickle them.
        """
        #q_fn = lambda x: np.quantile(x, q)
        exp = int(np.floor(np.log10(q)))
        man = int(q / 10 ** exp)

        return cls(f'quantile $1 - {man}\\times 10^{{{exp:1d}}}$', f"Q_1e{exp:3.1f}", '_quantile', 1.0-q)

    @classmethod
    def max(cls):
        #q_fn = lambda x: np.max(x)
        return cls('maximum', 'max', '_max',  1.0)




@attrs.define
class IndicatorFn:
    indicator: Indicator
    times: np.array
    ind_values: List[float]  = attrs.Factory(list)
    _spline: Any= None

    @staticmethod
    def common_max_time(ind_functions: List['IndicatorFn']) -> float:
        # Determine time index coles to the average max time of the indicators.
        max_times = [ind.time_max()[0]  for ind in ind_functions]
        avg_time = np.mean( max_times )
        idx = (np.abs(ind_functions[0].times - avg_time)).argmin()
        return idx

    def append(self, field_values):
        """
        Apply indicator reduction operation and append to values.
        """
        self.ind_values.append(self.indicator.reduction(field_values))

    @property
    def spline(self):
        if self._spline is None:
            self._spline = InterpolatedUnivariateSpline(self.times, self.ind_values, k=3)
        return self._spline

    def times_fine(self):
        return np.linspace(self.times[0], self.times[-1], num=len(self.times) * 10, endpoint=True)

    def time_max(self) -> (float, float):
        """
        :return: time of the max value, max value
        """
        times = self.times_fine()
        fine_values = self.spline(times)
        itime = np.argmax(fine_values)
        return times[itime], fine_values[itime]

def indicator_set():
    return [
        Indicator.quantile(0.005),
        Indicator.quantile(0.002),
        Indicator.quantile(0.001),
        Indicator.quantile(0.0005),
        Indicator.max()
    ]


#@memoize
@report
def indicators(pvd_in : File, attr_name, z_loc) -> List[IndicatorFn]:
    #extractor = Extractor.from_point_data(attr_name, z_loc)
    extractor = Extractor.from_cell_data(attr_name, z_loc)
    pvd_content = pv.get_reader(pvd_in.path)
    times = np.asarray(pvd_content.time_values)
    #print(times)

    indicators = indicator_set()
    ind_functions = [IndicatorFn(ind, times) for ind in indicators]

    for i, t in enumerate(times):
        pvd_content.set_active_time_point(i)
        dataset = pvd_content.read()
        values = extractor(dataset)
        for i in ind_functions:
            i.append(values)

    return ind_functions

#
# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return idx, array[idx]
#
#
#
# def quadratic_spline_roots(spl):
#     roots = []
#     knots = spl.get_knots()
#     for a, b in zip(knots[:-1], knots[1:]):
#         u, v, w = spl(a), spl((a+b)/2), spl(b)
#         t = np.roots([u+w-2*v, w-u, 2*v])
#         t = t[np.isreal(t) & (np.abs(t) <= 1)]
#         roots.extend(t*(b-a)/2 + (b+a)/2)
#     return np.array(roots)
#
# def getmax(spline):
#     cr_pts = quadratic_spline_roots(f.derivative())
#     cr_pts = np.append(cr_pts, (t[0], t[-1]))
#     cr_vals = spline(cr_pts)
#     max_index = np.argmax(cr_vals)
#     return cr_pts[max_index], cr_vals[max_index]
#
#
# def slice_plot(mesh, slices):
#     cmap = plt.cm.get_cmap("viridis", 4)
#     p = pv.Plotter()
#     p.add_mesh(mesh.outline(), color='k')
#     for s in slices:
#         p.add_mesh(s, cmap=cmap)
#     p.show()
#
# def time_plot
