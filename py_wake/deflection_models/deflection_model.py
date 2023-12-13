from abc import ABC, abstractmethod
from py_wake.utils.model_utils import method_args
from py_wake import np
from py_wake.utils import gradients
from numpy import newaxis as na


class DeflectionModel(ABC):

    @property
    def args4model(self):
        return self.args4deflection

    @property
    def args4deflection(self):
        return method_args(self.calc_deflection)

    @abstractmethod
    def calc_deflection(self, dw_ijl, hcw_ijl, dh_ijl, **kwargs):
        """Calculate deflection

        This method must be overridden by subclass

        Arguments required by this method must be added to the class list
        args4deflection

        See documentation of EngineeringWindFarmModel for a list of available input arguments

        Returns
        -------
        dw_ijlk : array_like
            downwind distance from source wind turbine(i) to destination wind turbine/site (j)
            for all wind direction (l) and wind speed (k)
        hcw_ijlk : array_like
            horizontal crosswind distance from source wind turbine(i) to destination wind turbine/site (j)
            for all wind direction (l) and wind speed (k)
        dh_ijlk : array_like
            vertical distance from source wind turbine(i) to destination wind turbine/site (j)
            for all wind direction (l) and wind speed (k)
        """


class DeflectionIntegrator(DeflectionModel):

    def __init__(self, N):
        self.N = N

    @property
    def args4deflection(self):
        return (DeflectionModel.args4deflection.fget(self) |  # @UndefinedVariable
                set(method_args(self.get_deflection_rate)) - {'theta_ilk', 'dw_ijlkx'})

    def calc_deflection(self, dw_ijlk, hcw_ijlk, dh_ijlk, yaw_ilk, tilt_ilk, **kwargs):
        dw_lst = (np.logspace(0, 1.1, self.N) - 1) / (10**1.1 - 1)
        dw_ijlkx = dw_ijlk[..., na] * dw_lst[na, na, na, na, :]

        theta_yaw_ilk, theta_tilt_ilk = gradients.deg2rad(yaw_ilk), gradients.deg2rad(-tilt_ilk)

        # alternative formulation
        # L = np.hypot(1, np.tan(theta_yaw_ilk))
        # theta_ilk = np.arctan(gradients.hypot(np.tan(theta_yaw_ilk), np.tan(theta_tilt_ilk) * L))
        # theta_deflection_ilk = gradients.arctan2(np.tan(theta_tilt_ilk) * L, np.tan(theta_yaw_ilk))

        theta_total_ilk = np.arcsin(gradients.hypot(np.sin(theta_yaw_ilk) * np.cos(theta_tilt_ilk),
                                                    np.sin(theta_tilt_ilk)))
        theta_total_angle_ilk = gradients.arctan2(np.sin(theta_tilt_ilk),
                                                  np.sin(theta_yaw_ilk) * np.cos(theta_tilt_ilk))

        deflection_rate = self.get_deflection_rate(theta_ilk=theta_total_ilk, dw_ijlkx=dw_ijlkx,
                                                   yaw_ilk=yaw_ilk, tilt_ilk=tilt_ilk, **kwargs)
        deflection_ijlk = gradients.trapz(deflection_rate, dw_ijlkx, axis=4)
        self.hcw_ijlk = hcw_ijlk + np.sign(dw_ijlk) * deflection_ijlk * np.cos(theta_total_angle_ilk[:, na])
        self.dh_ijlk = dh_ijlk + np.sign(dw_ijlk) * deflection_ijlk * np.sin(theta_total_angle_ilk[:, na])
        return dw_ijlk, self.hcw_ijlk, self.dh_ijlk

    @abstractmethod
    def get_deflection_rate(self):
        ""
