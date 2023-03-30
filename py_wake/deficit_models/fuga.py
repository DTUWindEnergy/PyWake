from numpy import newaxis as na
import xarray as xr
from py_wake import np
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel
from py_wake.superposition_models import LinearSum
from py_wake.tests.test_files import tfp
from py_wake.utils.fuga_utils import FugaUtils, LUTInterpolator
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from scipy.interpolate import RectBivariateSpline
from py_wake.utils import fuga_utils
from py_wake.utils.gradients import cabs
from py_wake.utils.grid_interpolator import GridInterpolator


class FugaDeficit(WakeDeficitModel, BlockageDeficitModel, FugaUtils):

    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc', remove_wriggles=False,
                 method='linear', rotorAvgModel=None, groundModel=None):
        """
        Parameters
        ----------
        LUT_path : str
            Path to folder containing 'CaseData.bin', input parameter file (*.par) and loop-up tables
        remove_wriggles : bool
            The current Fuga loop-up tables have significan wriggles.
            If True, all deficit values after the first zero crossing (when going from the center line
            and out in the lateral direction) is set to zero.
            This means that all speed-up regions are also removed
        """
        BlockageDeficitModel.__init__(self, upstream_only=True, rotorAvgModel=rotorAvgModel, groundModel=groundModel)
        FugaUtils.__init__(self, LUT_path, on_mismatch='input_par')
        self.remove_wriggles = remove_wriggles
        x, y, z, du = self.load()
        err_msg = "Method must be 'linear' or 'spline'. Spline is supports only height level only"
        assert method == 'linear' or (method == 'spline' and len(z) == 1), err_msg

        if method == 'linear':
            self.lut_interpolator = LUTInterpolator(x, y, z, du)
        else:
            du_interpolator = RectBivariateSpline(x, y, du[0].T)

            def interp(xyz):
                x, y, z = xyz
                assert np.all(z == self.z[0]), f'LUT table contains z={self.z} only'
                return du_interpolator.ev(x, y)
            self.lut_interpolator = interp

    def load(self):
        du = self.init_lut(self.load_luts(['UL'])[0], self.zHub, smooth2zero_x=None, smooth2zero_y=None,
                           remove_wriggles=self.remove_wriggles)
        return self.x, self.y, self.z, du

    def interpolate(self, x, y, z):
        # self.grid_interplator(np.array([zyx.flatten() for zyx in [z, y, x]]).T, check_bounds=False).reshape(x.shape)
        return self.lut_interpolator((x, y, z))

    def _calc_layout_terms(self, dw_ijlk, hcw_ijlk, z_ijlk, D_src_il, **_):
        self.mdu_ijlk = self.interpolate(dw_ijlk, cabs(hcw_ijlk), z_ijlk)

    def calc_deficit(self, WS_ilk, WS_eff_ilk, dw_ijlk, hcw_ijlk, z_ijlk, ct_ilk, D_src_il, **kwargs):
        if not self.deficit_initalized:
            self._calc_layout_terms(dw_ijlk, hcw_ijlk, z_ijlk, D_src_il, **kwargs)
        return self.mdu_ijlk * (ct_ilk * WS_eff_ilk**2 / WS_ilk)[:, na]

    def wake_radius(self, D_src_il, dw_ijlk, **_):
        # Set at twice the source radius for now
        return np.zeros_like(dw_ijlk) + D_src_il[:, na, :, na]


class FugaYawDeficit(FugaDeficit):

    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc',
                 remove_wriggles=False, method='linear', rotorAvgModel=None, groundModel=None):
        """
        Parameters
        ----------
        LUT_path : str
            Path to folder containing 'CaseData.bin', input parameter file (*.par) and loop-up tables
        remove_wriggles : bool
            The current Fuga loop-up tables have significan wriggles.
            If True, all deficit values after the first zero crossing (when going from the center line
            and out in the lateral direction) is set to zero.
            This means that all speed-up regions are also removed
        """
        BlockageDeficitModel.__init__(self, upstream_only=True, rotorAvgModel=rotorAvgModel, groundModel=groundModel)
        FugaUtils.__init__(self, LUT_path, on_mismatch='input_par')
        self.remove_wriggles = remove_wriggles
        x, y, z, dUL = self.load()

        mdUT = self.load_luts(['UT'])[0]
        dUT = np.array(mdUT, dtype=np.float32) * self.zeta0_factor(self.zHub)
        dU = np.concatenate([dUL[:, :, :, na], dUT[:, :, :, na]], 3)
        err_msg = "Method must be 'linear' or 'spline'. Spline is supports only height level only"
        assert method == 'linear' or (method == 'spline' and len(z) == 1), err_msg

        if method == 'linear':
            self.lut_interpolator = LUTInterpolator(x, y, z, dU)
        else:
            UL_interpolator = RectBivariateSpline(x, y, dU[0, :, :, 0].T)
            UT_interpolator = RectBivariateSpline(x, y, dU[0, :, :, 1].T)

            def interp(xyz):
                x, y, z = xyz
                assert np.all(z == self.z[0]), f'LUT table contains z={self.z} only'
                return np.moveaxis([UL_interpolator.ev(x, y), UT_interpolator.ev(x, y)], 0, -1)
            self.lut_interpolator = interp

    def _calc_layout_terms(self, dw_ijlk, hcw_ijlk, z_ijlk, D_src_il, **_):
        self.mdu_ijlk = (self.interpolate(dw_ijlk, cabs(hcw_ijlk), z_ijlk))

    def calc_deficit_downwind(self, WS_ilk, WS_eff_ilk, dw_ijlk, hcw_ijlk,
                              z_ijlk, ct_ilk, D_src_il, yaw_ilk, **_):

        mdUL_ijlk, mdUT_ijlk = np.moveaxis(self.interpolate(
            dw_ijlk, cabs(hcw_ijlk), z_ijlk), -1, 0)
        mdUT_ijlk = np.negative(mdUT_ijlk, out=mdUT_ijlk, where=hcw_ijlk < 0)  # UT is antisymmetric
        theta_ilk = np.deg2rad(yaw_ilk)

        mdu_ijlk = (mdUL_ijlk * np.cos(theta_ilk)[:, na] - mdUT_ijlk * np.sin(theta_ilk)[:, na])
        # avoid wake on itself
        mdu_ijlk *= ~((dw_ijlk == 0) & (hcw_ijlk <= D_src_il[:, na, :, na]))

        return mdu_ijlk * (ct_ilk * WS_eff_ilk**2 / WS_ilk)[:, na]

    def calc_deficit(self, **kwargs):
        # fuga result is already downwind
        return self.calc_deficit_downwind(**kwargs)


class Fuga(PropagateDownwind):
    def __init__(self, LUT_path, site, windTurbines,
                 rotorAvgModel=None, deflectionModel=None, turbulenceModel=None, remove_wriggles=False):
        """
        Parameters
        ----------
        LUT_path : str
            path to look up tables
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        PropagateDownwind.__init__(self, site, windTurbines,
                                   wake_deficitModel=FugaDeficit(LUT_path, remove_wriggles=remove_wriggles),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)


class FugaBlockage(All2AllIterative):
    def __init__(self, LUT_path, site, windTurbines, rotorAvgModel=None,
                 deflectionModel=None, turbulenceModel=None, convergence_tolerance=1e-6, remove_wriggles=False):
        """
        Parameters
        ----------
        LUT_path : str
            path to look up tables
        site : Site
            Site object
        windTurbines : WindTurbines
            WindTurbines object representing the wake generating wind turbines
        rotorAvgModel : RotorAvgModel, optional
            Model defining one or more points at the down stream rotors to
            calculate the rotor average wind speeds from.\n
            if None, default, the wind speed at the rotor center is used
        deflectionModel : DeflectionModel
            Model describing the deflection of the wake due to yaw misalignment, sheared inflow, etc.
        turbulenceModel : TurbulenceModel
            Model describing the amount of added turbulence in the wake
        """
        fuga_deficit = FugaDeficit(LUT_path, remove_wriggles=remove_wriggles)
        All2AllIterative.__init__(self, site, windTurbines, wake_deficitModel=fuga_deficit,
                                  rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                  deflectionModel=deflectionModel, blockage_deficitModel=fuga_deficit,
                                  turbulenceModel=turbulenceModel, convergence_tolerance=convergence_tolerance)


class list_indexer:
    def __init__(self, lst):
        self.lst = lst

    def __call__(self, x):
        return np.searchsorted(self.lst, np.minimum(x, self.lst[-1]))


class FugaMultiLUTDeficit(FugaDeficit):
    def __init__(self, LUT_path_lst=tfp + 'fuga/*.nc', remove_wriggles=False,
                 method='linear', rotorAvgModel=None, groundModel=None):
        BlockageDeficitModel.__init__(self, upstream_only=True, rotorAvgModel=rotorAvgModel, groundModel=groundModel)

        import glob

        def open_dataset(f):
            ds = xr.open_dataset(f).transpose('z', 'y', 'x')
            ds['TI'] = fuga_utils.ti(ds.z0, ds.hubheight)
            return ds

        if isinstance(LUT_path_lst, str):
            ds_lst = [open_dataset(f) for f in glob.glob(LUT_path_lst)]
        else:
            ds_lst = [open_dataset(f) for f in LUT_path_lst]

        x_lst, y_lst, z_lst = [ds_lst[0][k].values for k in 'xyz']
        assert np.all([np.all(ds.x == ds_lst[0].x) for ds in ds_lst])
        assert np.all([np.all(ds.y == ds_lst[0].y) for ds in ds_lst])
        assert np.all([np.all(ds.z == ds_lst[0].z) for ds in ds_lst])
        assert np.all([np.all(ds.z0 == ds_lst[0].z0) for ds in ds_lst])
        assert np.all([np.all(ds.zeta0 == ds_lst[0].zeta0) for ds in ds_lst])

        self.x = ds_lst[0].x.values
        self.y = ds_lst[0].y.values
        self.z = ds_lst[0].z.values
        self.z0 = ds_lst[0].z0.item()
        self.zeta0 = ds_lst[0].zeta0.item()

        data = np.concatenate([self.init_lut(ds.UL.values, ds.hubheight.item(), remove_wriggles=remove_wriggles)[na]
                               for ds in ds_lst], 0)

        i_lst = np.arange(len(ds_lst))
        self.interpolator = GridInterpolator([i_lst, z_lst, y_lst, x_lst], data,
                                             method=['nearest', 'linear', 'linear', 'linear'])

        d_lst = np.sort(np.unique([ds.diameter.item() for ds in ds_lst]))
        h_lst = np.sort(np.unique([ds.hubheight.item() for ds in ds_lst]))
        # ti_lst = np.sort(np.unique([ds.TI.item() for ds in ds_lst]))
        d_index, h_index = [list_indexer(lst) for lst in [d_lst, h_lst]]
        # ti_searcher =

        index_arr = np.full((len(d_lst), len(h_lst)), -1)
        for i, ds in enumerate(ds_lst):
            index_arr[d_index(ds.diameter.item()), h_index(ds.hubheight.item())] = i
        self.index_arr = index_arr
        self.d_index, self.h_index = d_index, h_index

    def _calc_layout_terms(self, dw_ijlk, hcw_ijlk, z_ijlk, D_src_il, h_ilk, **kwargs):
        i_ilk = self.index_arr[self.d_index(D_src_il)[:, :, na], self.h_index(h_ilk)]
        i_ijlk = np.broadcast_to(i_ilk[:, na, :], dw_ijlk.shape)
        xp = np.array([i_ijlk, z_ijlk, cabs(hcw_ijlk), dw_ijlk])
        self.mdu_ijlk = self.interpolator(xp.reshape((4, -1)).T, bounds='limit').reshape(dw_ijlk.shape)

        self.mdu_ijlk *= ~((dw_ijlk == 0) & (hcw_ijlk <= D_src_il[:, na, :, na]))  # avoid wake on itself


def main():
    if __name__ == '__main__':
        from py_wake.examples.data.iea37._iea37 import IEA37Site
        from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
        import matplotlib.pyplot as plt

        # setup site, turbines and wind farm model
        site = IEA37Site(16)
        x, y = site.initial_position.T
        windTurbines = IEA37_WindTurbines()

        path = tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc'

        for wf_model in [Fuga(path, site, windTurbines),
                         FugaBlockage(path, site, windTurbines)]:
            plt.figure()
            print(wf_model)

            # run wind farm simulation
            sim_res = wf_model(x, y)

            # calculate AEP
            aep = sim_res.aep().sum()

            # plot wake map
            flow_map = sim_res.flow_map(wd=30, ws=9.8)
            flow_map.plot_wake_map()
            flow_map.plot_windturbines()
            plt.title('AEP: %.2f GWh' % aep)
        plt.show()


main()
