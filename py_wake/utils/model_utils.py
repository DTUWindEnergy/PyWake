import inspect
import os
import pkgutil
import py_wake
from pathlib import Path
import numpy as np
from numpy import newaxis as na


def get_exclude_dict():
    from py_wake.deficit_models.deficit_model import DeficitModel, ConvectionDeficitModel, WakeDeficitModel,\
        BlockageDeficitModel
    from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel, RotorCenter
    from py_wake.wind_farm_models.wind_farm_model import WindFarmModel
    from py_wake.wind_farm_models.engineering_models import EngineeringWindFarmModel, PropagateDownwind

    from py_wake.superposition_models import SuperpositionModel, LinearSum
    from py_wake.deflection_models.deflection_model import DeflectionModel
    from py_wake.turbulence_models.turbulence_model import TurbulenceModel
    from py_wake.ground_models import GroundModel
    from py_wake.deficit_models.noj import NOJDeficit
    from py_wake.ground_models.ground_models import NoGround
    return {
        WindFarmModel: ([EngineeringWindFarmModel], [], PropagateDownwind),
        DeficitModel: ([ConvectionDeficitModel, BlockageDeficitModel, WakeDeficitModel], [RotorAvgModel], NOJDeficit),
        WakeDeficitModel: ([ConvectionDeficitModel], [RotorAvgModel], NOJDeficit),
        RotorAvgModel: ([], [], RotorCenter),
        SuperpositionModel: ([], [], LinearSum),
        BlockageDeficitModel: ([], [], None),
        DeflectionModel: ([], [], None),
        TurbulenceModel: ([], [], None),
        GroundModel: ([], [], NoGround)

    }


def cls_name(cls):
    if cls is None:
        return "None"
    return cls.__name__


def cls_in(A, cls_lst):
    def path(c):
        return str(Path(inspect.getsourcefile(c)).resolve())

    pywake_path = str(Path(py_wake.__file__).parent.resolve())
    pywake_classes = [c for c in cls_lst
                      if c is not object and cls_name(c) == cls_name(A) and
                      path(c).startswith(pywake_path)]

    return any([path(A) in [path(c) for c in pywake_classes]])


def get_models(base_class):

    exclude_cls_lst, exclude_subcls_lst, default = get_exclude_dict()[base_class]

    model_lst = []
    for loader, module_name, _ in pkgutil.walk_packages([os.path.dirname(inspect.getabsfile(base_class))]):
        if 'test' in module_name:
            continue
        _module = loader.find_module(module_name).load_module(module_name)
        for n in dir(_module):
            v = _module.__dict__[n]
            if inspect.isclass(v):
                if (cls_in(base_class, v.mro()) and
                    not cls_in(v, exclude_cls_lst + [base_class]) and
                    not any([issubclass(v, cls) for cls in exclude_subcls_lst]) and
                        not cls_in(v, model_lst)):
                    model_lst.append(v)

    if default is not None:
        model_lst.remove(model_lst[[cls_name(m) for m in model_lst].index(cls_name(default))])
    model_lst.insert(0, default)
    return model_lst


def list_models():
    for model_type in list(get_exclude_dict().keys()):
        print("%s (from %s import *)" % (model_type.__name__, ".".join(model_type.__module__.split(".")[:2])))
        for model in get_models(model_type):
            if model is not None:
                print("\t%s%s" % (model.__name__, str(inspect.signature(model.__init__)).replace('self, ', '')))


def get_signature(cls, kwargs={}, indent_level=0):
    sig = inspect.signature(cls.__init__)

    def get_arg(n, arg_value):
        if arg_value is None:
            arg_value = sig.parameters[n].default
            if 'object at' in str(arg_value):
                arg_value = get_signature(arg_value.__class__, indent_level=(indent_level + 1, 0)[indent_level == 0])
            elif isinstance(arg_value, str):
                arg_value = "'%s'" % arg_value
        else:
            arg_value = get_signature(arg_value, indent_level=(indent_level + 1, 0)[indent_level == 0])
        if arg_value is inspect._empty:
            return n
        if isinstance(arg_value, np.ndarray):
            arg_value = arg_value.tolist()
        return "%s=%s" % (n, arg_value)
    if indent_level:
        join_str = ",\n%s" % (" " * 4 * indent_level)
    else:
        join_str = ", "
    arg_str = join_str.join([get_arg(n, kwargs.get(n, None))
                             for n in sig.parameters if n not in {'self', 'args', 'kwargs'}])
    if indent_level and arg_str:
        return "%s(%s%s)" % (cls.__name__, join_str[1:], arg_str)
    else:
        return "%s(%s)" % (cls.__name__, arg_str)


def get_model_input(wfm, x, y, ws=10, wd=270, yaw=[[[0]]], tilt=[[[0]]]):
    ws, wd = [np.atleast_1d(v) for v in [ws, wd]]
    x, y = map(np.asarray, [x, y])
    wfm.site.distance.setup(src_x_i=[0], src_y_i=[0], src_h_i=[0],
                            dst_xyh_j=(x, y, x * 0))
    dw_ijl, hcw_ijl, dh_ijl = wfm.site.distance(wd_il=wd[na])
    sim_res = wfm([0], [0], ws=ws, wd=wd, yaw=yaw)

    args = {'dw_ijl': dw_ijl, 'hcw_ijl': hcw_ijl, 'dh_ijl': dh_ijl,
            'D_src_il': np.atleast_1d(wfm.windTurbines.diameter())[na]}
    args.update({k: sim_res[n].ilk() for k, n in [('yaw_ilk', 'yaw'),
                                                  ('tilt_ilk', 'tilt'),
                                                  ('WS_ilk', 'WS'),
                                                  ('WS_eff_ilk', 'WS_eff'),
                                                  ('ct_ilk', 'CT')]})
    return args


def check_model(model, cls, arg_name=None, accept_None=True):
    if not isinstance(model, cls):
        if model is None and accept_None:
            return

        if arg_name is not None:
            s = f'Argument, {arg_name}, '
        else:
            s = f'{model} '
        s += f'must be a {cls.__name__} instance'
        if inspect.isclass(model) and issubclass(model, cls):
            raise ValueError(s + f'. Did you forget the brackets: {model.__name__}()')

        raise ValueError(s + f', but is a {model.__class__.__name__} instance')


def fix_shape(arr, shape_or_arr_to_match, allow_number=False, allow_None=False):
    if allow_None and arr is None:
        return arr
    if allow_number and isinstance(arr, (int, float)):
        return arr

    arr = np.asarray(arr)
    if isinstance(shape_or_arr_to_match, tuple):
        shape = shape_or_arr_to_match
    else:
        shape = np.asarray(shape_or_arr_to_match).shape
    return np.broadcast_to(arr.reshape(arr.shape + (1,) * (len(shape) - len(arr.shape))), shape)


def main():
    if __name__ == '__main__':
        list_models()


main()
