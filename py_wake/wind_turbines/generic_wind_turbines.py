from py_wake.wind_turbines._wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import SimpleYawModel, PowerCtTabular, PowerCtNDTabular
import numpy as np
from py_wake.utils.generic_power_ct_curves import standard_power_ct_curve


class GenericWindTurbine(WindTurbine):
    def __init__(self, name, diameter, hub_height, power_norm, turbulence_intensity=.1,
                 air_density=1.225, max_cp=.49, constant_ct=.8,
                 gear_loss_const=.01, gear_loss_var=.014, generator_loss=0.03, converter_loss=.03,
                 ws_lst=np.arange(.1, 30, .1), ws_cutin=None,
                 ws_cutout=None, power_idle=0, ct_idle=None, method='linear',
                 additional_models=[SimpleYawModel()]):
        """Wind turbine with generic standard power curve based on max_cp, rated power and losses.
        Ct is computed from the basic 1d momentum theory

        Parameters
        ----------
        name : str
            Wind turbine name
        diameter : int or float
            Diameter of wind turbine
        power_norm : int or float
            Nominal power [kW]
        diameter : int or float
            Rotor diameter [m]
        turbulence_intensity : float
            Turbulence intensity
        air_density : float optional
            Density of air [kg/m^3], defualt is 1.225
        max_cp : float
            Maximum power coefficient
        constant_ct : float, optional
            Ct value in constant-ct region
        gear_loss_const : float
            Constant gear loss [%]
        gear_loss_var : float
            Variable gear loss [%]
        generator_loss : float
            Generator loss [%]
        converter_loss : float
            converter loss [%]
        ws_lst : array_like
            List of wind speeds. The power/ct tabular will be calculated for these wind speeds
        ws_cutin : number or None, optional
            if number, then the range [0,ws_cutin[ will be set to power_idle and ct_idle
        ws_cutout : number or None, optional
            if number, then the range ]ws_cutout,100] will be set to power_idle and ct_idle
        power_idle : number, optional
            see ws_cutin and ws_cutout
        ct_idle : number, optional
            see ws_cutin and ws_cutout
        method : {'linear', 'phip','spline}
            Interpolation method:\n
            - linear: fast, discontinous gradients\n
            - pchip: smooth\n
            - spline: smooth, closer to linear, small overshoots in transition to/from constant plateaus)
        additional_models : list, optional
            list of additional models.
        """
        u, p, ct = standard_power_ct_curve(power_norm, diameter, turbulence_intensity, air_density, max_cp, constant_ct,
                                           gear_loss_const, gear_loss_var, generator_loss, converter_loss, ws_lst)
        if ws_cutin is not None:
            u, p, ct = [v[u >= ws_cutin] for v in [u, p, ct]]
        if ws_cutout is not None:
            u, p, ct = [v[u <= ws_cutout] for v in [u, p, ct]]
        if ct_idle is None:
            ct_idle = ct[-1]
        powerCtFunction = PowerCtTabular(u, p * 1000, 'w', ct, ws_cutin=ws_cutin, ws_cutout=ws_cutout,
                                         power_idle=power_idle, ct_idle=ct_idle, method=method,
                                         additional_models=additional_models)
        WindTurbine.__init__(self, name, diameter, hub_height, powerCtFunction)


class GenericTIRhoWindTurbine(WindTurbine):
    def __init__(self, name, diameter, hub_height, power_norm,
                 TI_eff_lst=np.linspace(0, .5, 6), default_TI_eff=.1,
                 Air_density_lst=np.linspace(.9, 1.5, 5), default_Air_density=1.225,
                 max_cp=.49, constant_ct=.8,
                 gear_loss_const=.01, gear_loss_var=.014, generator_loss=0.03, converter_loss=.03,
                 wsp_lst=np.arange(.1, 30, .5),
                 additional_models=[SimpleYawModel()]):
        """Wind turbine with generic standard power curve based on max_cp, rated power and losses.
        Ct is computed from the basic 1d momentum theory
        The power and ct curves depends on turbulence intensity(TI_eff) and air density(Air_density)

        Parameters
        ----------
        name : str
            Wind turbine name
        diameter : int or float
            Diameter of wind turbine
        power_norm : int or float
            Nominal power [kW]
        diameter : int or float
            Rotor diameter [m]
        TI_eff_lst : array_like
            List of turbulence intensities to include in tabular
        default_TI_eff : float, optional
            Default turbulence intensity, default is 10%
        Air_density_lst : array_like
            List of air densities [kg/m^3] to include in tabular
        default_Air_density : float, optional
            Default air_density [kg/m^3], defualt is 1.225
        max_cp : float
            Maximum power coefficient
        constant_ct : float, optional
            Ct value in constant-ct region
        gear_loss_const : float
            Constant gear loss [%]
        gear_loss_var : float
            Variable gear loss [%]
        generator_loss : float
            Generator loss [%]
        converter_loss : float
            converter loss [%]
        additional_models : list, optional
            list of additional models.
        """

        data = np.moveaxis([[standard_power_ct_curve(power_norm, diameter, ti, rho, max_cp, constant_ct, gear_loss_const,
                                                     gear_loss_var, generator_loss, converter_loss, wsp_lst)[1:]
                             for rho in Air_density_lst]
                            for ti in TI_eff_lst], -1, 0)
        p = data[:, :, :, 0]
        ct = data[:, :, :, 1]

        powerCtFunction = PowerCtNDTabular(
            input_keys=['ws', 'TI_eff', 'Air_density'],
            value_lst=[wsp_lst, TI_eff_lst, Air_density_lst],
            power_arr=p * 1000, power_unit='w', ct_arr=ct,
            default_value_dict={k: v for k, v in [('TI_eff', default_TI_eff), ('Air_density', default_Air_density)]
                                if v is not None},
            additional_models=additional_models)
        WindTurbine.__init__(self, name, diameter, hub_height, powerCtFunction)
