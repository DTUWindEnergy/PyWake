.. _installation:

===========================
Installation
===========================

You must first have a working and properly configured Python 3.4+ distribution
on your computer. We often use and highly recommend
`Anaconda <https://www.anaconda.com/download/>`_.

Simple user
------------

If you have other Python programs besides PyWake, it is a good idea to install
each program in its own environment to ensure that the dependencies for the
different packages do not conflict with one another. The commands to create and
then activate an environment in an Anaconda prompt are::

   conda create --name pywake python=3.6
   activate pywake

You can install PyWake into the current active environment directly from git::

   pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git


Developer
----------

We highly recommend developers install PyWake into its own environment. (See
instructions above.) The commands to clone and install PyWake with developer
options into the current active environment in an Anaconda Prommpt are as
follows::

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git
   cd PyWake
   pip install -e .

