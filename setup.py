# -*- coding: utf-8 -*-
"""
Setup file for PyWake
"""
import os
from setuptools import setup, find_packages
import pkg_resources

repo = os.path.dirname(__file__)
try:
    from git_utils import write_vers
    version = write_vers(vers_file='py_wake/__init__.py', repo=repo, skip_chars=1)
except Exception:
    version = '999'


try:
    from pypandoc import convert_file

    def read_md(f): return convert_file(f, 'rst', format='md')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")

    def read_md(f): return open(f, 'r').read()


setup(name='py_wake',
      version=version,
      description='PyWake a collection of wake models',
      long_description=read_md('README.md'),
      url='https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake',
      project_urls={
          'Documentation': 'https://topfarm.pages.windenergy.dtu.dk/PyWake/',
          'Changelog': 'https://topfarm.pages.windenergy.dtu.dk/PyWake/notebooks/ChangeLog.html',
          'Source': 'https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake',
          'Tracker': 'https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/-/issues',
      },
      author='DTU Wind Energy',
      author_email='mmpe@dtu.dk',
      license='MIT',
      packages=find_packages(),
      package_data={
          'py_wake': ['examples/data/iea37/*.yaml',
                      'examples/data/*.npz',
                      'tests/test_files/fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00/*.*',
                      'tests/test_files/fuga/2MW/Z0=0.00408599Zi=00400Zeta0=0.00E+00/*.*',
                      'tests/test_files/fuga/2MW/*.nc',
                      'examples/data/iea34_130rwt/*/*/*.h5',
                      'examples/data/iea34_130rwt/*/*/*.json',
                      'examples/data/ParqueFicticio/*.grd',
                      'examples/data/NEG-Micon-2750.wtg',
                      'examples/data/Vestas V112-3.0 MW.wtg',
                      'rotor_avg_models/gaussian_overlap_.02_.02_128_512.nc'
                      ],
      },
      # When adding extra requirements be sure to update recipe/meta.yaml
      install_requires=[
          'matplotlib',  # for plotting
          'numpy',  # for numerical calculations
          'xarray', 'netcdf4', 'h5netcdf',
          'autograd',  # gradient calculation
          'pyyaml',  # for reading yaml files
          'scipy',  # constraints
          'tqdm',  # progressbar
      ],
      extras_require={
          'test': [
              'pytest',  # for testing
              'pytest-cov',  # for calculating coverage
              'psutil',  # memory profiling
              'memory_profiler',  # memory profiling
              'sphinx',  # generating documentation
              'sphinx_rtd_theme',  # docs theme
              'line_profiler',  # to check speed
              'scikit-learn',  # MinMaxScaler
              'tensorflow',  # load surrogates
              'ipywidgets',  # notebook widgets
          ],
          'loads': [
              'scikit-learn',  # MinMaxScaler
              'tensorflow',  # load surrogates
          ]},
      zip_safe=True)
