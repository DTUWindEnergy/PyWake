from pathlib import Path
import os


def svg2eps(f, replace=lambda s: s.replace(".svg", ".eps")):
    txt = replace(f.read_text(encoding='utf-8')).replace("\r", "")
    with open(f, 'w', encoding='utf-8', newline='\n') as fid:
        fid.write(txt)


if __name__ == '__main__':
    import py_wake
    docs_path = Path(py_wake.__file__).parents[1] / 'docs'
    if os.path.isdir(docs_path / 'api'):
        print("Switch to PDF mode")
        for ext in ['*.rst', 'notebooks/*.ipynb']:
            for f in docs_path.glob(ext):
                svg2eps(f)
        (docs_path / 'api').rename("api_hide")
    else:
        print("Switch to html mode")
        for ext in ['*.rst', 'notebooks/*.ipynb']:
            for f in docs_path.glob(ext):
                svg2eps(f, lambda s: s.replace('.eps', '.svg'))
        (docs_path / 'api_hide').rename("api")
