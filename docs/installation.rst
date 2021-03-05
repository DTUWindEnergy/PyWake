


Installation
===========================

.. toctree::
    :maxdepth: 2

    install_python
    

    

Install PyWake (Simple user)
----------------------------

* Install from PyPi.org (official releases)::
  
    pip install py_wake

* Install from gitlab  (includes any recent updates)::
  
    pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git
        


Install PyWake (Developer)
--------------------------

We highly recommend developers install PyWake into its own environment. (See
instructions above.) The commands to clone and install PyWake with developer
options including dependencies required to run the tests into the current active 
environment in an Anaconda Prommpt are as follows::

   git clone https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git
   cd PyWake
   pip install -e .[test]
   


