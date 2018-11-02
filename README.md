[![pipeline status](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/badges/master/pipeline.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/commits/master)
[![coverage report](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/badges/master/coverage.svg)](https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake/commits/master)

# PyWake

This is a work-in-progress attempt to make something like FUSEDWAKE 2.0. Its relation to FUSEDWAKE is, however, not determined yet.
The idea is that it should:

- Reduce duplicated code
- Include wakemap functions
- Be faster
- Be suitable for layout optimization
- Support use of load surrrogate models
- Support complex terrain
- Support gradient based optimization

## Installation

Get the code from git:

    $ git clone https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake.git

Make sure that you are using Python 3.4 or higher. The script can be 
installed using:

    $ cd PyWake
    $ python setup.py develop

## Run Horns Rev 1 example

To calculate the annual energy production of Horns Rev 1, go to:

    $ cd py_wake/examples

and run:

    $ python hornsrev1_example.py




