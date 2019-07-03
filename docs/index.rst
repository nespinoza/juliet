juliet
======

`juliet` is a versatile modelling tool for transiting and non-transiting exoplanetary 
systems that allows to perform quick-and-easy fits to data coming from transit photometry, 
radial velocity or both using bayesian inference and, in particular, using Nested Sampling in 
order to allow both efficient fitting and proper model comparison.

In this documentation you'll be able to check out the features `juliet` can offer for your 
research, which range from fitting different datasets simultaneously for both transits and 
radial-velocities to accounting for systematic trends both using linear models or 
Gaussian Processes (GP), to even extract information from photometry alone (e.g., stellar rotation 
periods) with just a few lines of code.

`juliet` builds on the work of "giants" that have made publicly available tools for transit (`batman <https://www.cfa.harvard.edu/~lkreidberg/batman/>`_, 
`starry <https://rodluger.github.io/starry/>`_), radial-velocity (`radvel <https://radvel.readthedocs.io/en/latest/>`_), GP modelling 
(`george <https://george.readthedocs.io/en/latest/>`_, `celerite <https://celerite.readthedocs.io/en/stable/>`_) and Nested Samplers (`MultiNest` via 
`pymultinest <https://github.com/JohannesBuchner/PyMultiNest>`_, `dynesty <https://dynesty.readthedocs.io>`_)  and thus can be seen as a wrapper of all 
of those in one. Somewhat like an `Infinity Gauntlet <https://cdn.shopify.com/s/files/1/0882/5118/products/Infinity-Gauntlet-by-Jim-Starlin-1306917_1024x1024.jpeg?v=1438791299>`_ 
for exoplanets.

`juliet` is in active development in its `public repository on GitHub
<https://github.com/nespinoza/juliet>`_. If you discover any bugs or have requests for us, please consider 
sending us an email or `opening an issue <https://github.com/nespinoza/juliet/issues>`_.

.. image:: https://img.shields.io/badge/GitHub-nespinoza%2Fjuliet-blue.svg?style=flat
    :target: https://github.com/nespinoza/juliet
.. image:: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
    :target: https://github.com/nespinoza/juliet/LICENSE

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user/installation
   user/quicktest

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/transitfits
   tutorials/rvfits
   tutorials/jointfits
   tutorials/linearmodels
   tutorials/gps
   tutorials/multithreading


Contributors
------------

.. include:: ../AUTHORS.rst


License & Attribution
---------------------

Copyright 2018-2019 Nestor Espinoza & Diana Kossakowski.

`juliet` is being developed by `Nestor Espinoza <http://www.nestor-espinoza.com>`_ and Diana Kossakowski in a
`public GitHub repository <https://github.com/nespinoza/juliet>`_. The source code is made available under the 
terms of the MIT license.

If you make use of this code, please cite `the paper which is currently on the
ArXiv <https://arxiv.org/abs/1812.08549>`_:

.. code-block:: tex

    @ARTICLE{juliet,
           author = {{Espinoza}, N{\'e}stor and {Kossakowski}, Diana and {Brahm}, Rafael},
            title = "{Juliet: a versatile modelling tool for transiting and non-transiting exoplanetary systems}",
          journal = {arXiv e-prints},
         keywords = {Astrophysics - Earth and Planetary Astrophysics},
             year = 2018,
            month = Dec,
              eid = {arXiv:1812.08549},
            pages = {arXiv:1812.08549},
    archivePrefix = {arXiv},
           eprint = {1812.08549},
     primaryClass = {astro-ph.EP},
           adsurl = {https://ui.adsabs.harvard.edu/\#abs/2018arXiv181208549E},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }
