

Install Python
==============

    For all platforms we recommend that you download and install the Anaconda - 
    a professional grade, full blown scientific Python distribution.

    Installing Anaconda, activate root environment:
    
        * Download and install Anaconda (Python 3.x version, 64 bit installer is recommended) from https://www.continuum.io/downloads
        
        * Update the root Anaconda environment (type in a terminal): 
            
            ``>> conda update --all``
        
        * Activate the Anaconda root environment in a terminal as follows: 
            
            ``>> activate``
            
Create envirronment
===================

    If you have other Python programs besides PyWake, it is a good idea to install
    each program in its own environment to ensure that the dependencies for the
    different packages do not conflict with one another. The commands to create and
    then activate an environment in an Anaconda prompt are::
    
       conda create --name pywake python=3.6
       activate pywake