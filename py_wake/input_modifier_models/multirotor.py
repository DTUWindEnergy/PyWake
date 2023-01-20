from py_wake.input_modifier_models.input_modifier_model import InputModifierModel
import numpy as np
from py_wake.examples.data.hornsrev1 import V80
from py_wake.wind_turbines._wind_turbines import WindTurbines
from numpy import newaxis as na
from py_wake.wind_farm_models.engineering_models import All2AllIterative
from py_wake.utils.model_utils import check_model


class MultiRotorWindTurbines(WindTurbines):

    def __init__(self, windTurbine, offsets=[[-50, 30, 0], [50, 30, 0]]):
        """Instantiate a multirotor wind turbines object

        Parameters
        ----------
        windTurbine : WindTurbine object
            windturbine to base the model on
        offsets : array_like (no_rotors, (x,y,z))
            offset of each rotor. y is downwind, z up, and z according to the right-hand rule
        """
        check_model(windTurbine, WindTurbines)
        wt = windTurbine
        n, o, d, h, pct = zip(*[(f'{wt.name()}_{i}', o, wt.diameter(), wt.hub_height(), wt.powerCtFunction)
                                for i, o in enumerate(offsets)])
        WindTurbines.__init__(self, names=n, diameters=d, hub_heights=h, powerCtFunctions=pct)
        self.offset = np.asarray(o)

    def position(self, x, y, h, types, wd):

        x, y, h = [np.expand_dims(v, tuple(range(1, 3 - len(np.shape(v)) + 1))) for v in [x, y, h]]
        x_offset, y_offset, z_offset = [v[:, na, na] for v in np.atleast_2d(self._info(self.offset, types)).T]

        theta = ((90 - wd) / 180 * np.pi)[na, :, na]
        return (x + x_offset * np.sin(theta) - y_offset * np.cos(theta),
                y - x_offset * np.cos(theta) - y_offset * np.sin(theta),
                h + z_offset)


class MultiRotor(InputModifierModel):
    def setup(self, x_ilk, y_ilk, h_ilk, wd, type_i, **_):
        return {k: v for k, v in zip(['x_ilk', 'y_ilk', 'h_ilk'],
                                     self.windFarmModel.windTurbines.position(x_ilk, y_ilk, h_ilk, type_i, wd))}


def main():
    if __name__ == '__main__':

        import matplotlib.pyplot as plt
        from py_wake.deficit_models.noj import NOJDeficit
        from py_wake.site.xrsite import UniformSite

        wts = MultiRotorWindTurbines(V80())

        wfm = All2AllIterative(site=UniformSite(), windTurbines=wts, wake_deficitModel=NOJDeficit(),
                               inputModifierModels=MultiRotor())

        sim_res = wfm([0, 0], [0, 0], wd=[0, 90, 180, 270], ws=[10, 11, 12, 16], type=[0, 1])
        for ax, wd in zip(plt.subplots(2, 2)[1].flatten(), sim_res.wd):
            sim_res.flow_map(wd=wd).plot_wake_map(ax=ax)
        plt.show()


main()
