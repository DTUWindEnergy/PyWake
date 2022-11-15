if __name__ == '__main__':  # pragma: no cover
    from pathlib import Path
    from pyfuga import get_luts
    import inspect
    import os
    import shutil
    import numpy as np
    from py_wake.utils import fuga_utils
    path = Path(inspect.getsourcefile(get_luts)).parent.parent
    for zhub, d, z0 in [(70, 80, 1e-5),
                        # (70, 80, .12),
                        # (70, 80, .18),
                        (90, 120, 1e-5),
                        # (90, 120, .18)
                        ]:
        # z0 = np.round(fuga_utils.z0(ti, zhub), 8)
        print(zhub, d, z0)
        luts = get_luts(folder=path, zeta0=0, nkz0=16, nbeta=32,
                        diameter=d, zhub=zhub, z0=z0, zi=400, zlow=68, zhigh=90,
                        lut_vars=['UL'], nx=512, ny=128,)
        shutil.copy(path / (luts.name + ".nc"), Path(__file__).parent)
