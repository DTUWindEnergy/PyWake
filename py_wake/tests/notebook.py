import json
import os
from os.path import dirname
from os.path import join as pjoin
import re
import ssl
import sys
import matplotlib.pyplot as plt
from _io import StringIO


class Notebook():
    pip_header = """# Install PyWake if needed
try:
    import py_wake
except ModuleNotFoundError:
    !pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git"""

    def __init__(self, filename):
        self.filename = filename
        try:
            self.nb = self.load_notebook(self.filename)
        except Exception as e:
            raise Exception('Error in ', os.path.relpath(filename)) from e

    def __repr__(self):
        return "hej"

    def load_notebook(self, filename):
        with open(filename, encoding='utf-8') as fid:
            nb = json.load(fid)
        return nb

    def save(self, filename=None):
        filename = filename or self.filename
        with open(filename, 'w') as fid:
            json.dump(self.nb, fid, indent=4)

    def __getitem__(self, key):
        return self.nb[key]

    def __setitem__(self, key, value):
        self.nb[key] = value

    def __getattribute__(self, name):
        try:
            return object.__getattribute__(self, name)
        except AttributeError:
            if name in self.nb.keys():
                return self.nb[name]
            raise

    def insert_markdown_cell(self, index, text):
        self.cells.insert(index, {"cell_type": "markdown",
                                  "metadata": {},
                                  "source": [l + "\n" for l in text.split("\n")]
                                  })

    def insert_code_cell(self, index, code):
        self.cells.insert(index,
                          {"cell_type": "code",
                           "execution_count": 0,
                           "metadata": {},
                              "outputs": [],
                              "source": code,
                           })

    def replace_include_tag(self):
        cells = []
        for cell in self.nb['cells']:
            if cell['cell_type'] == 'code' and len(cell['source']) > 0 and '%%include' in cell['source'][0]:
                filename = pjoin(dirname(self.filename), cell['source'][0].replace('%%include', '').strip())
                nb = Notebook(filename)
                nb.replace_include_tag()
                cells.extend(nb.cells)
            else:
                cells.append(cell)
        return cells

    def get_code(self):
        code = []
        for cell in self.cells:
            if cell['cell_type'] == "code":
                if "".join(cell['source']).strip() != "":
                    code.append("".join(cell['source']))
        return code

    def get_text(self):
        txt = []
        for cell in self.cells:
            if cell['cell_type'] == "markdown":
                if "".join(cell['source']).strip() != "":
                    txt.append("".join(cell['source']))
        return txt

    def check_code(self):
        code = "\n".join(self.get_code())

        def fix(line):
            for p in ['%', '!']:
                if line.strip().startswith(p):
                    line = line.replace(p, "pass #")
            return line

        lines = [fix(l) for l in code.split("\n")]

        # import * only allowed at module level
        # So extract and remove from code lines
        module_imports = [l for l in lines if l.startswith('from') and l.endswith('import *')]
        for l in module_imports:
            lines.remove(l)

        if len(lines) == 1 and lines[0] == '':
            return
        try:
            import contextlib

            with contextlib.redirect_stdout(StringIO()):
                with contextlib.redirect_stderr(StringIO()):
                    # execute module level imports (stored in l dict) and use a locals in execution of code
                    g, l = {}, {}
                    exec("\n".join(module_imports), g, l)

                    code_str = "def test():\n    " + "\n    ".join(lines) + "\ntest()"
                    exec(code_str, l, {})
                    plt.close('all')
        except Exception as e:
            raise type(e)("Code error in %s\n%s\n" % (self.filename, str(e))).with_traceback(sys.exc_info()[2])

    def check_links(self):
        txt = "\n".join(self.get_text())
        for link in re.finditer(r"\[([^]]*)]\(([^)]*)\)", txt):
            label, url = link.groups()
            # print(label)
            # print(url)
            if url.startswith('attachment') or '#' in url:
                continue
            if url.startswith("../_static"):
                assert os.path.isfile(os.path.join(os.path.dirname(self.filename), url))
                return

            try:
                import urllib.request
                context = ssl._create_unverified_context()
                assert urllib.request.urlopen(url, context=context).getcode() == 200
            except Exception as e:
                print("%s broken in %s\n%s" % (url, self.filename, str(e)))

                # traceback.print_exc()

        # print(txt)

    def check_pip_header(self):

        code = self.get_code()
        if not code:
            return
        if code[0].strip() != self.pip_header:
            for i, cell in enumerate(self.cells):
                if cell['cell_type'] == "code":
                    break
            self.insert_code_cell(i, self.pip_header)
            self.save()
            raise Exception("""pip install header was not present in %s.
It has now been auto insert. Please check the notebook and commit the changes""" % os.path.abspath(self.filename))

    def remove_empty_end_cell(self):
        while self.cells[-1]['cell_type'] == 'code' and all([l.strip() == "" for l in self.cells[-1]['source']]):
            self.nb['cells'] = self.cells[:-1]
            self.save()


if __name__ == '__main__':
    import py_wake
    nb = Notebook(os.path.dirname(py_wake.__file__) + '/../docs/notebooks/QuickStart.ipynb')
    nb.check_code()
    nb.check_links()
    nb.remove_empty_end_cell()
    nb.check_pip_header()
