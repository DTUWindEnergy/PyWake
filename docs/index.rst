.. PyWake documentation master file, created by
   sphinx-quickstart on Mon Dec  3 13:24:21 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyWake
===========================================

.. image:: logo.svg
    :width: 70 %
    :align: center

PyWake is an open-sourced and Python-based wind farm simulation tool developed at DTU capable of computing flow fields, power production of individual turbines as well as the Annual Energy Production (AEP) of a wind farm. The software solution provides an interface to both a selection different engineering models as well as CDF RANS (PyWakeEllipSys). It is highly efficient in calculating how the wake propagates within a wind farm and can quantify the interaction between turbines.

What can PyWake do?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main objective of PyWake is to calculate the wake interaction in a wind farm in a computationally inexpensive way for a range of steady state conditions. It is very useful for computing the power production of a wind farm while considering the wake losses for a specific wind farm layout configuration. Some of the main capabilities of PyWake that have been in constant development in the last few years include:

    * The possibility to use different engineering wake models for the simulation, such as the NOJ and Bastankhah wake deficit models.
    * The option of choosing between different sites and their wind resource, with the additional option of user-defined sites.
    * The ability to have user-defined wind turbines or import turbine files from WAsP.
    * The capability of working with chunkification and parallelization.
    * The advantage of visualizing flow maps for the wind farm layout in study.

For installation instructions, please see the :ref:`Installation Guide <installation>`.

Source code repository and issue tracker:
    https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake
    
License:
    MIT_

.. _MIT: https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/blob/master/LICENSE

Getting Started
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
PyWake is equipped with many capabilities that can range from basic to complex. For new users, the :ref:`Overview </notebooks/Overview.ipynb>` section contains a basic description of PyWake’s architecture and the elements behind it. Plus, the :ref:`Quickstart </notebooks/Quickstart.ipynb>` section shows how to set up and perform some basic operations in PyWake.

Explanations of PyWake's core objects can be found in the following tutorials:

	* :ref:`Site </notebooks/Site.ipynb>`: this tutorial walks through the set up of pre-defined sites in PyWake as well as the possibility for user-defined sites.
	* :ref:`Wind Turbine </notebooks/WindTurbines.ipynb>`: this example demonstrates how to set up a wind turbine object and also to create user-defined turbines with specific power and CT curves.
	* :ref:`Engineering Wind Farm Models </notebooks/EngineeringWindFarmModels.ipynb>`: here there is a detailed explanation of all the wake deficit models available in PyWake as well as other engineering models such as wake superposition, deflection, etc.

The :ref:`Wind farm simulation </notebooks/RunWindFarmSimulation.ipynb>` example shows how to execute PyWake and extract relevant information about the wind farm studied. In addition, PyWake's capablities to calculate gradients are demonstrated in the :ref:`Gradients, parallelization and precision </notebooks/gradients_parallellization.ipynb>` example, and an optimization with TOPFARM is available in the :ref:`Optimization </notebooks/Optimization.ipynb>` tutorial.

Lastly, the remaining notebooks illustrate some relevant examples and exercises to see the different properties that PyWake has to offer.


    .. toctree::
        :maxdepth: 1
	:caption: Contents
    
        installation
        notebooks/Overview  
        notebooks/ChangeLog
        notebooks/Publications
               
    .. toctree::
        :maxdepth: 1
	:caption: Tutorials
       
	notebooks/Quickstart
        notebooks/Site
        notebooks/WindTurbines
        notebooks/EngineeringWindFarmModels
	notebooks/RunWindFarmSimulation
	notebooks/gradients_parallellization
        notebooks/Optimization
        notebooks/YawMisalignment
        notebooks/exercises/CombineModels
        notebooks/exercises/Validation
        notebooks/exercises/Improve_layout
        notebooks/exercises/WakeDeflection  

    .. toctree::
        :maxdepth: 1
	:caption: Model Verification       

        notebooks/literature_verification/TurbOPark
        
    .. toctree::
        :maxdepth: 2
	:caption: Validation
    
        validation
        
    .. toctree::
        :maxdepth: 1
	:caption: API Reference
            
        api/WindTurbines
        api/Site
        api/WindFarmModel
        api/EngineeringWindFarmModels
        api/PredefinedEngineeringWindFarmModels
        api/SimulationResult
        api/FlowMap
        

