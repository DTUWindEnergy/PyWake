from numpy import newaxis as na
import xarray as xr
from py_wake import np
from py_wake.deficit_models.deficit_model import WakeDeficitModel, BlockageDeficitModel, XRLUTDeficitModel
from py_wake.superposition_models import LinearSum
from py_wake.tests.test_files import tfp
from py_wake.utils.fuga_utils import FugaUtils, LUTInterpolator, FugaXRLUT
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from scipy.interpolate import RectBivariateSpline
from py_wake.utils import fuga_utils
from py_wake.utils.gradients import cabs
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.utils.model_utils import DeprecatedModel
import glob
from xarray.core.merge import merge_attrs


class FugaDeficit(WakeDeficitModel, BlockageDeficitModel, FugaUtils):

    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc',
                 smooth2zero_x=None, smooth2zero_y=None, remove_wriggles=False,
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
        smooth2zero_x : int or None, optional
            make a linear transition to zero over the first and last <smoot2zero_x> points in the downwind direction
            of the look-up table
            if None, default, smooth2zero_x is set to 1/8 of the box length
            if 0, no correction is applied.
            if >0, the first and last <smooth2zero_x> points are linearly faded to zero
        smooth2zero_y : int or None, optional
            make a linear transition to last <smoot2zero_y> points in the cross wind direction
            of the look-up table
            if None, default, smooth2zero_y is set to 1/8 of the box width (i.e. center line to the side)
            if 0, no correction is applied.
            if >0, the <smooth2zero_x> points farthest away from the centerline are linearly faded to zero
        """
        BlockageDeficitModel.__init__(self, upstream_only=True, rotorAvgModel=rotorAvgModel, groundModel=groundModel)
        FugaUtils.__init__(self, LUT_path, on_mismatch='input_par')
        self.smooth2zero_x = smooth2zero_x
        self.smooth2zero_y = smooth2zero_y
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

        X, Y = np.meshgrid(self.x, self.y)
        self.lut_interpolator([X, Y, X * 0 + self.zHub])
        du_zhub = self.lut_interpolator([X, Y, X * 0 + self.zHub])
        self.setup_wake_radius(du_zhub)

    def setup_wake_radius(self, du_zhub):
        # set wake limit center_deficit * np.exp(-2)), corresponding to value of 2 sigma for a gaussian profile
        wake_radius_arr = self.y[np.argmin((du_zhub > (du_zhub[0] * np.exp(-2))), 0)]

        rp = len(self.x) // 4
        wake_radius_arr[:rp] = 0  # set upstream to 0
        self.wake_radius_arr = wake_radius_arr

    def load(self):
        du = self.init_lut(self.load_luts(['UL'])[0],
                           smooth2zero_x=self.smooth2zero_x, smooth2zero_y=self.smooth2zero_y,
                           remove_wriggles=self.remove_wriggles)
        return self.x, self.y, self.z, du

    def interpolate(self, x, y, z):
        # self.grid_interplator(np.array([zyx.flatten() for zyx in [z, y, x]]).T, check_bounds=False).reshape(x.shape)
        return self.lut_interpolator((x, y, z))

    def _calc_layout_terms(self, dw_ijlk, hcw_ijlk, z_ijlk, D_src_il, **_):
        self.mdu_ijlk = self.interpolate(dw_ijlk, cabs(hcw_ijlk), z_ijlk)

    def calc_deficit(self, WS_ilk, WS_eff_ilk, dw_ijlk, hcw_ijlk, z_ijlk, ct_ilk, D_src_il, **kwargs):
        if not self.deficit_initalized:
            self._calc_layout_terms(dw_ijlk=dw_ijlk, hcw_ijlk=hcw_ijlk, z_ijlk=z_ijlk, D_src_il=D_src_il, **kwargs)
        return self.mdu_ijlk * (ct_ilk * WS_eff_ilk**2 / WS_ilk)[:, na]

    def wake_radius(self, D_src_il, dw_ijlk, **_):
        return np.interp(dw_ijlk, self.x, self.wake_radius_arr)


class FugaYawDeficit(FugaDeficit):

    def __init__(self, LUT_path=tfp + 'fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00.nc',
                 smooth2zero_x=None, smooth2zero_y=None, remove_wriggles=False,
                 method='linear', rotorAvgModel=None, groundModel=None):
        """
        Parameters
        ----------
        LUT_path : str
            Path to folder containing 'CaseData.bin', input parameter file (*.par) and loop-up tables
        smooth2zero_x : int or None, optional
            make a linear transition to zero over the first and last <smoot2zero_x> points in the downwind direction
            of the look-up table
            if None, default, smooth2zero_x is set to 1/8 of the box length
            if 0, no correction is applied.
            if >0, the first and last <smooth2zero_x> points are linearly faded to zero
        smooth2zero_y : int or None, optional
            make a linear transition to last <smoot2zero_y> points in the cross wind direction
            of the look-up table
            if None, default, smooth2zero_y is set to 1/8 of the box width (i.e. center line to the side)
            if 0, no correction is applied.
            if >0, the <smooth2zero_x> points farthest away from the centerline are linearly faded to zero
        remove_wriggles : bool
            The current Fuga loop-up tables have significan wriggles.
            If True, all deficit values after the first zero crossing (when going from the center line
            and out in the lateral direction) is set to zero.
            This means that all speed-up regions are also removed
        """
        BlockageDeficitModel.__init__(self, upstream_only=True, rotorAvgModel=rotorAvgModel, groundModel=groundModel)
        FugaUtils.__init__(self, LUT_path, on_mismatch='input_par')
        self.smooth2zero_x = smooth2zero_x
        self.smooth2zero_y = smooth2zero_y
        self.remove_wriggles = remove_wriggles
        x, y, z, dUL = self.load()

        mdUT = self.load_luts(['UT'])[0]
        dUT = np.array(mdUT, dtype=np.float32) * self.zeta0_factor()
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
        X, Y = np.meshgrid(self.x, self.y)
        self.lut_interpolator([X, Y, X * 0 + self.zHub])
        du_zhub = self.lut_interpolator([X, Y, X * 0 + self.zHub])[:, :, 0]
        self.setup_wake_radius(du_zhub)

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


class Fuga(PropagateDownwind, DeprecatedModel):
    def __init__(self, LUT_path, site, windTurbines,
                 rotorAvgModel=None, deflectionModel=None, turbulenceModel=None,
                 smooth2zero_x=None, smooth2zero_y=None, remove_wriggles=False):
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
                                   wake_deficitModel=FugaDeficit(LUT_path,
                                                                 smooth2zero_x=smooth2zero_x,
                                                                 smooth2zero_y=smooth2zero_y,
                                                                 remove_wriggles=remove_wriggles),
                                   rotorAvgModel=rotorAvgModel, superpositionModel=LinearSum(),
                                   deflectionModel=deflectionModel, turbulenceModel=turbulenceModel)
        DeprecatedModel.__init__(self, 'py_wake.literature.fuga.Ott_2014')


class FugaBlockage(All2AllIterative, DeprecatedModel):
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
        DeprecatedModel.__init__(self, 'py_wake.literature.fuga.Ott_2014_Blockage')


class FugaMultiLUTDeficit(XRLUTDeficitModel, FugaDeficit):
    def __init__(self, LUT_path_lst=tfp + 'fuga/2MW/multilut/LUTs_Zeta0=0.00e+00_16_32_*_zi400_z0=0.00001000_z9.8-207.9_UL_nx128_ny128_dx20.0_dy5.0.nc',
                 z_lst=None, TI_ref_height=None, bounds='limit',
                 smooth2zero_x=None, smooth2zero_y=None, remove_wriggles=False,
                 rotorAvgModel=None, groundModel=None,
                 use_effective_ti=False):

        fuga_kwargs = dict(smooth2zero_x=smooth2zero_x, smooth2zero_y=smooth2zero_y, remove_wriggles=remove_wriggles)
        if isinstance(LUT_path_lst, str):
            da_lst = [FugaXRLUT(f, **fuga_kwargs).UL for f in glob.glob(LUT_path_lst)]
            assert len(da_lst), f"No files found matching {LUT_path_lst}"
        else:
            da_lst = [FugaXRLUT(f, **fuga_kwargs).UL for f in LUT_path_lst]

        dims = ['d_h', 'zeta0', 'zi', 'z0']

        da_lst = [da.assign_coords({'d_h': da.diameter * 1000 + da.hubheight,
                                    **{k: getattr(da, k) for k in dims[1:]}}).expand_dims(dims) for da in da_lst]
        self.TI_ref_height = TI_ref_height

        if z_lst is None:
            z_lst = np.sort(np.unique([da.z for da in da_lst]))
        x_lst = np.sort(np.unique([da.x for da in da_lst]))
        y_lst = np.sort(np.unique([da.y for da in da_lst]))
        da_lst = [da.interp(z=z_lst, x=x_lst, y=y_lst) for da in da_lst]

        # combine_by_coords does not always merge attributes correctly
        attrs = merge_attrs([da.attrs for da in da_lst], combine_attrs='drop_conflicts')
        da = xr.combine_by_coords(da_lst, combine_attrs='drop').squeeze()
        da.attrs = attrs
        self.x, self.y = da.x.values, da.y.values
        self._args4model = {k + "_ilk" for k in ['zeta0', 'zi'] if k in da.dims}

        method = ['nearest'] + (['linear'] * (len(da.dims) - 1))
        XRLUTDeficitModel.__init__(self, da, get_input=self.get_input, method=method, bounds=bounds,
                                   rotorAvgModel=rotorAvgModel, groundModel=groundModel,
                                   use_effective_ws=False, use_effective_ti=use_effective_ti)

    def wake_radius(self, D_src_il, dw_ijlk, **_):
        # Set at twice the source radius for now
        return np.zeros_like(dw_ijlk) + D_src_il[:, na, :, na]

    def calc_deficit(self, WS_ilk, WS_eff_ilk, dw_ijlk, hcw_ijlk, z_ijlk, ct_ilk, D_src_il, **kwargs):
        # bypass XRLUTDeficitModel.calc_deficit
        return FugaDeficit.calc_deficit(self, WS_ilk=WS_ilk, WS_eff_ilk=WS_eff_ilk,
                                        dw_ijlk=dw_ijlk, hcw_ijlk=hcw_ijlk, z_ijlk=z_ijlk,
                                        ct_ilk=ct_ilk, D_src_il=D_src_il, **kwargs)

    def _calc_layout_terms(self, **kwargs):
        self.mdu_ijlk = XRLUTDeficitModel.calc_deficit(self, **kwargs)

    def get_input(self, D_src_il, TI_ilk, h_ilk, dw_ijlk, hcw_ijlk, z_ijlk, **kwargs):
        user = {'zeta0': lambda: kwargs['zeta0_ilk'][:, na],
                'zi': lambda: kwargs['zi_ilk'][:, na]}
        interp_kwargs = {'d_h': (D_src_il[:, :, na] * 1000 + h_ilk)[:, na],
                         'z0': fuga_utils.z0(TI_ilk, self.TI_ref_height or h_ilk, zeta0=kwargs.get('zeta0_ilk', 0))[:, na],
                         'z': z_ijlk,
                         'x': dw_ijlk,
                         'y': cabs(hcw_ijlk)}
        interp_kwargs.update({k: v() for k, v in user.items() if k in self.da.dims})
        return [interp_kwargs[k] for k in self.da.dims]

    def get_output(self, output_ijlk, **kwargs):
        return output_ijlk


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
