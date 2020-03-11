
Site classes
=================

.. inheritance-diagram:: py_wake.site.WaspGridSite
    :top-classes: py_wake.site._site.Site
    :parts: 1


- `Site`_: base class
- `UniformSite`_: Site with uniform (same wind over all, i.e. flat uniform terrain) and constant wind speed probability of 1. Only for one fixed wind speed
- `UniformWeibullSite`_: Site with uniform (same wind over all, i.e. flat uniform terrain) and weibull distributed wind speed
- `WaspGridSite`_: Site with non-uniform (different wind at different locations, e.g. complex non-flat terrain) weibull distributed wind speed. Data obtained from WAsP grid files


Site
-----------------

.. autoclass:: py_wake.site.Site
    :members:
       
    .. autosummary::
        distances
        elevation
        local_wind
        probability
        plot_ws_distribution
        plot_wd_distribution
        

    
    
UniformSite
-----------------

.. autoclass:: py_wake.site.UniformSite

       
    .. automethod:: __init__
    


UniformWeibullSite
------------------

.. autoclass:: py_wake.site.UniformWeibullSite

    .. automethod:: __init__
    
    

WaspGridSite
-----------------

.. autoclass:: py_wake.site.WaspGridSite

       
    .. automethod:: __init__
    
    
