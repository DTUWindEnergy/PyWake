from py_wake import np
from numpy import newaxis as na
from py_wake.deficit_models import DeficitModel
from py_wake.deficit_models import BlockageDeficitModel
from py_wake.utils.gradients import hypot
from py_wake.deficit_models.utils import a0
import warnings


class RankineHalfBody(BlockageDeficitModel):
    """
    A simple induction model using a Rankine Half Body to represent the induction
    introduced by a wind turbine. The source strength is determined enforcing 1D
    momentum balance at the rotor disc.
    References:
        [1] B Gribben, G Hawkes - A potential flow model for wind turbine
            induction and wind farm blockage - Technical Paper, Frazer-Nash Consultancy, 2019
    """

    def __init__(self, limiter=1e-10, exclude_wake=True, superpositionModel=None,
                 rotorAvgModel=None, groundModel=None, upstream_only=False):
        BlockageDeficitModel.__init__(self, upstream_only=upstream_only, superpositionModel=superpositionModel,
                                      rotorAvgModel=rotorAvgModel, groundModel=groundModel)
        # limiter to avoid singularities
        self.limiter = limiter
        # if used in a wind farm simulation, set deficit in wake region to
        # zero, as here the wake model is active
        self.exclude_wake = exclude_wake

    def outside_body(self, WS_ilk, a0_ilk, R_il, dw_ijlk, cw_ijlk, r_ijlk):
        """
        Find all points lying outside Rankine Half Body, stagnation line given on p.3
        """
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', r'invalid value encountered in true_divide')
            warnings.filterwarnings('ignore', r'invalid value encountered in divide')
            cos_ijlk = dw_ijlk / r_ijlk
        # avoid division by zero
        f_ilk = a0_ilk * R_il[:, :, na]
        f_ilk = np.where(f_ilk == 0., np.inf, f_ilk)
        # replaced sin**2 and m of expression given in [1]
        val = cos_ijlk - 1 / f_ilk[:, na] * cw_ijlk**2
        iout = val <= -1.
        return iout

    def calc_deficit(self, WS_ilk, D_src_il, dw_ijlk, cw_ijlk, ct_ilk, **_):
        # source strength as given on p.7
        a0_ilk = a0(ct_ilk)
        R_il = D_src_il / 2.
        m_ilk = 2. * WS_ilk * a0_ilk * np.pi * R_il[:, :, na]**2
        # radial distance
        r_ijlk = hypot(dw_ijlk, cw_ijlk)
        # find points lying outside RHB, the only ones to be computed
        # remove singularities
        iout = self.outside_body(*(np.real(v) for v in [WS_ilk, a0_ilk, R_il, dw_ijlk, cw_ijlk, r_ijlk]))
        # deficit, p.3 equation for u, negative to get deficit

        deficit_ijlk = np.where(((2 * r_ijlk / D_src_il[:, na, :, na]) < self.limiter) | ~iout, 0,
                                -m_ilk[:, na] / (4 * np.pi) * dw_ijlk / (r_ijlk + (r_ijlk == 0))**3)

        if self.exclude_wake:
            deficit_ijlk = self.remove_wake(deficit_ijlk, dw_ijlk, cw_ijlk, D_src_il)
            # Close to the rotor the induced velocities become unphysical and are
            # limited to the induction in the rotor plane estimated by BEM.
            induc = (WS_ilk * a0(ct_ilk))[:, na]
            deficit_ijlk = np.where(deficit_ijlk > induc, induc * np.sign(deficit_ijlk), deficit_ijlk)

        return deficit_ijlk


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        from py_wake.superposition_models import LinearSum
        from py_wake.wind_farm_models import All2AllIterative
        from py_wake.deficit_models.no_wake import NoWakeDeficit
        import matplotlib.pyplot as plt

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()
        rhb = RankineHalfBody()

        plt.figure()
        noj_rhb = All2AllIterative(site, windTurbines, wake_deficitModel=NoWakeDeficit(),
                                   superpositionModel=LinearSum(), blockage_deficitModel=rhb)
        flow_map = noj_rhb(x=[0], y=[0], wd=[270], ws=[10]).flow_map()
        clevels = np.array([.6, .7, .8, .9, .95, .98, .99, .995, .998, .999, 1., 1.01, 1.02]) * 10.
        flow_map.plot_wake_map(levels=clevels)
        plt.title('Rankine Half Body')
        plt.ylabel("Crosswind distance [y/R]")
        plt.xlabel("Downwind distance [x/R]")
        plt.show()

        # run wind farm simulation
        sim_res = noj_rhb(x, y, wd=[0, 30, 45, 60, 90], ws=[5, 10, 15])

        # calculate AEP
        aep = sim_res.aep().sum()

        # plot wake map
        plt.figure()
        print(noj_rhb)
        flow_map = sim_res.flow_map(wd=0, ws=10)
        flow_map.plot_wake_map(levels=clevels, plot_colorbar=False)
        plt.title('Rankine Half Body, AEP: %.3f GWh' % aep)
        plt.show()


main()
