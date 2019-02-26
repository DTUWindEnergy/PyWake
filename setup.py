# -*- coding: utf-8 -*-
"""
Setup file for PyWake
"""
import os
from git_utils import write_vers
from setuptools import setup, find_packages

repo = os.path.dirname(__file__)
version = write_vers(vers_file='py_wake/__init__.py', repo=repo, skip_chars=1)

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
      author='DTU Wind Energy',
      author_email='mmpe@dtu.dk',
      license='MIT',
      packages=find_packages(),
          package_data={
          'py_wake': ['examples/data/iea37/*.yaml',
                      'tests/test_files/fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+0/*.*'],
      },
      install_requires=[
          'matplotlib',  # for plotting
          'numpy',  # for numerical calculations
          'xarray',  # for WaspGridSite data storage
          'pytest',  # for testing
          'pytest-cov',  # for calculating coverage
          'pyyaml',  # for reading yaml files
          'scipy',  # constraints
          'sphinx',  # generating documentation
          'sphinx_rtd_theme'  # docs theme
      ],
      zip_safe=True)
