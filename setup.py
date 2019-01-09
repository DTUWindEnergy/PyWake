# -*- coding: utf-8 -*-
"""
Setup file for PyWake
"""
from py_wake import __version__

from setuptools import setup, find_packages

try:
    from pypandoc import convert_file
    read_md = lambda f: convert_file(f, 'rst', format='md')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()


setup(name='py_wake',
      version=__version__,
      description='PyWake a collection of wake models',
      long_description=read_md('README.md'),
      url='https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake',
      author='DTU Wind Energy',
      author_email='mmpe@dtu.dk',
      license='MIT',
      packages=find_packages(),
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
