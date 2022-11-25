.. _installation:

Installation Guide
===========================


Pre-Installation
----------------------------
Before you can install the software, you must first have a working Python distribution with a package manager. For all platforms we recommend that you download and install Anaconda - a professional grade, full-blown scientific Python distribution.

To set up Anaconda, you should:

    * Download and install Anaconda (Python 3.x version, 64 bit installer is recommended) from https://www.continuum.io/downloads
    
    * Update the root Anaconda environment (type in a terminal): 
        
        ``>> conda update --all``
    
    * Activate the Anaconda root environment in a terminal as follows: 
        
        ``>> activate``

It is recommended to create a new environment to install PyWake if you have other Python programs. This ensures that the dependencies for the different packages do not conflict with one another. In the command prompt, create and active an environment with::

   conda create --name pywake python=3.8
   activate pywake


Simple Installation
----------------------------

PyWake’s base code is open-sourced and freely available on `GitLab 
<https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake>`_ (MIT license).

* Install from PyPi.org (official releases)::
  
    pip install py_wake

* Install from GitLab  (includes any recent updates)::
  
    pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git
        


Developer Installation
-------------------------------

We highly recommend developers to install PyWake into the environment created previously. The commands to clone and install PyWake with developer options including dependencies required to run the tests into the current active environment in an Anaconda Prompt are as follows::

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git
   cd PyWake
   pip install -e .[test]