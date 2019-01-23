import os
import json
import pprint
import shutil
from _notebooks.notebook import Notebook


# def get_cells(nb):
#     cells = []
#     for cell in nb['cells']:
#         if cell['cell_type'] == 'code' and len(cell['source']) > 0 and '%%include' in cell['source'][0]:
#             cells.extend(load_notebook(cell['source'][0].replace('%%include', '').strip())['cells'])
#         else:
#             cells.append(cell)
#     return cells

#
# def load_notebook(f):
#     with open(f) as fid:
#         nb = json.load(fid)
#
#     nb['cells'] = get_cells(nb)
#     return nb
#
#
# def save_notebook(nb, f):
#     with open(f, 'w') as fid:
#         json.dump(nb, fid, indent=4)
#         # fid.write(pprint.pformat(nb))


def make_tutorials():
    path = os.path.dirname(__file__) + "/templates/"
    for f in [f for f in os.listdir(path) if f.endswith('.ipynb')]:
        nb = Notebook(path + f)
        nb.replace_include_tag()
        nb.save(os.path.dirname(__file__) + "/../tutorials/" + f)
        # with open(os.path.dirname(__file__) + "/../tutorials/" + f, 'w') as fid:
        #    json.dump(nb, fid)


def doc_header(name):
    nb = Notebook(os.path.dirname(__file__) + "/elements/doc_setup.ipynb")
    nb.cells[0]['source'][0] = nb.cells[0]['source'][0].replace('[name]', name)
    return nb.cells


def make_doc_notebooks(notebooks):
    src_path = os.path.dirname(__file__) + "/elements/"
    dst_path = os.path.dirname(__file__) + "/../docs/notebooks/"
    if os.path.isdir(dst_path):
        try:
            shutil.rmtree(dst_path)

        except PermissionError:
            pass
    os.makedirs(dst_path, exist_ok=True)
    for name in notebooks:
        nb = Notebook(src_path + name + ".ipynb")
        t = '[Try this yourself](https://colab.research.google.com/github/DTUWindEnergy/PyWake/blob/master/docs/notebooks/%s.ipynb) (requires google account)'
        nb.insert_markdown_cell(1, t % name)
        code = """%%capture
# Install PyWake if needed
try:
    import py_wake
except ModuleNotFoundError:
    !pip install py_wake"""
        nb.insert_code_cell(2, code)
        #cells = nb.cells

        #cells = cells[:1] + doc_header(name) + cells[1:]
        #nb['cells'] = cells
        nb.save(dst_path + name + ".ipynb")
#         with open(dst_path + name + ".ipynb", 'w') as fid:
#             json.dump(nb, fid)


def check_notebooks():
    import matplotlib.pyplot as plt

    def no_show(*args, **kwargs):
        pass
    plt.show = no_show  # disable plt show that requires the user to close the plot

    path = os.path.dirname(__file__) + "/elements/"
    for f in [f for f in os.listdir(path) if f.endswith('.ipynb')]:
        nb = Notebook(path + f)
        nb.check_code()
        nb.check_links()


if __name__ == '__main__':
    check_notebooks()
    make_tutorials()
    make_doc_notebooks(['V80', 'IEA37Turbine',
                        'IEA37Site', 'Hornsrev1Site',
                        'noj', 'fuga', 'IEA37SimpleBastankhahGaussian', 'BastankhahGaussian',
                        'cmp_wakemodels',
                        'noj_validation_exercise_solution', 'noj_validation_exercise', 'hornsrev_layout_optimization_exercise'])
    print('Done')
