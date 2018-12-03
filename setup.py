# -*- coding: utf-8 -*-
"""
Setup file for PyWake
"""
from py_wake import __version__

from setuptools import setup

setup(name='py_wake',
      version=__version__,
      description='PyWake a collection of wake models',
      url='https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake',
      author='DTU Wind Energy',
      author_email='mmpe@dtu.dk',
      license='MIT',
      packages=['py_wake'
                ],
      install_requires=[
          'matplotlib',  # for plotting
          'numpy',  # for numerical calculations
          'pytest',  # for testing
          'pytest-cov',  # for calculating coverage
          'pyyaml',  # for reading yaml files
          'scipy',  # constraints
          'sphinx',  # generating documentation
          'sphinx_rtd_theme'  # docs theme
      ],
      zip_safe=True)
