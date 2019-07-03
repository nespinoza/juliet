.. _quicktest:

Non-transiting photometric fitting
===================

Two ways of using juliet
-------------------------

In the spirit of accomodating the code for everyone to use, `juliet` can be used in two different ways: as 
an *imported library* and also in *command line mode*. Both give rise to the same results because the command 
line mode simply calls the `juliet` libraries in a python script.

To use `juliet` as an *imported library*, inside any python script you can simply do:

.. code-block:: python

    import juliet
    out = juliet.fit(priors,t_lc=times,y_lc=flux,yerr_lc=flux_error)

In this example, `juliet` will perform a fit on a lightcurve defined by a vector of times `times`, 
relative fluxes `flux` and error on those fluxes `flux_error` given some prior information `prior` which, 
as we will see below, is defined through a dictionary. 


In *command line mode*, `juliet` can be used through a simple call in any terminal. To do this, after 
installing juliet, you can from anywhere in your system simply do:

.. code-block:: bash

    juliet -flag1 -flag2 --flag3

In this example, `juliet` is performing a fit using different inputs defined by `-flag1`, `-flag2` and `--flag3`. 
There are several flags that can be used to accomodate your `juliet` runs. If this mode suits your needs, 
check out the `project's wiki page to find out more about this mode <https://github.com/nespinoza/juliet/wiki>`_.

A primer on transit and radial-velocity fitting
-----------------------------------------------

As an example on transit and radial-velocity fitting, here we perform a fit to data 
