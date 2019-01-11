import os
import json
import pprint
import shutil


def get_cells(nb):
    cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and len(cell['source']) > 0 and '%%include' in cell['source'][0]:
            cells.extend(load_notebook(cell['source'][0].replace('%%include', '').strip())['cells'])
        else:
            cells.append(cell)
    return cells


def load_notebook(f):
    with open(f) as fid:
        nb = json.load(fid)

    nb['cells'] = get_cells(nb)
    return nb


def save_notebook(nb, f):
    with open(f, 'w') as fid:
        json.dump(nb, fid, indent=4)
        # fid.write(pprint.pformat(nb))


def make_tutorials():
    path = os.path.dirname(__file__) + "/templates/"
    for f in [f for f in os.listdir(path) if f.endswith('.ipynb')]:
        nb = load_notebook(path + f)
        save_notebook(nb, os.path.dirname(__file__) + "/../tutorials/" + f)
        # with open(os.path.dirname(__file__) + "/../tutorials/" + f, 'w') as fid:
        #    json.dump(nb, fid)


def doc_header(name):
    nb = load_notebook(os.path.dirname(__file__) + "/elements/doc_header.ipynb")
    nb['cells'][0]['source'][0] = nb['cells'][0]['source'][0].replace('[name]', name)
    return nb['cells']


def make_doc_notebooks(notebooks):
    src_path = os.path.dirname(__file__) + "/elements/"
    dst_path = os.path.dirname(__file__) + "/../docs/notebooks/"
    if os.path.isdir(dst_path):
        shutil.rmtree(dst_path)
    os.makedirs(dst_path)
    for name in notebooks:
        nb = load_notebook(src_path + name + ".ipynb")
        cells = nb['cells']

        cells = cells[:1] + doc_header(name) + cells[1:]
        nb['cells'] = cells
        save_notebook(nb, dst_path + name + ".ipynb")
#         with open(dst_path + name + ".ipynb", 'w') as fid:
#             json.dump(nb, fid)


if __name__ == '__main__':
    make_tutorials()
    make_doc_notebooks(['V80', 'IEA37Turbine', 'IEA37Site', 'HornsrevSite', 'noj', 'fuga'])
    print('Done')
