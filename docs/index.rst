juliet
======

.. image:: ../juliet.png
    :width: 400pt
    :align: center

``juliet`` is a versatile modelling tool for transiting and non-transiting exoplanetary 
systems that allows to perform quick-and-easy fits to data coming from transit photometry, 
radial velocity or both using bayesian inference and, in particular, using Nested Sampling in 
order to allow both efficient fitting and proper model comparison.

In this documentation you'll be able to check out the features ``juliet`` can offer for your 
research, which range from fitting different datasets simultaneously for both transits and 
radial-velocities to accounting for systematic trends both using linear models or 
Gaussian Processes (GP), to even extract information from photometry alone (e.g., stellar rotation 
periods) with just a few lines of code.

``juliet`` builds on the work of "giants" that have made publicly available tools for transit (`batman <https://www.cfa.harvard.edu/~lkreidberg/batman/>`_, 
`starry <https://rodluger.github.io/starry/>`_), radial-velocity (`radvel <https://radvel.readthedocs.io/en/latest/>`_), GP modelling 
(`george <https://george.readthedocs.io/en/latest/>`_, `celerite <https://celerite.readthedocs.io/en/stable/>`_) and Nested Samplers (`MultiNest` via 
`pymultinest <https://github.com/JohannesBuchner/PyMultiNest>`_, `dynesty <https://dynesty.readthedocs.io>`_, `ultranest <https://johannesbuchner.github.io/UltraNest/>`_)  and thus can be seen as a wrapper of all 
of those in one. Somewhat like an `Infinity Gauntlet <https://cdn.shopify.com/s/files/1/0882/5118/products/Infinity-Gauntlet-by-Jim-Starlin-1306917_1024x1024.jpeg?v=1438791299>`_ 
for exoplanets.

The library is in active development in its `public repository on GitHub
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
   user/priorsnparameters
   user/api

.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/transitfits
   tutorials/rvfits
   tutorials/jointfits
   tutorials/linearmodels
   tutorials/gps
   tutorials/ttvs
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

If you make use of this code, please cite `the paper <https://arxiv.org/abs/1812.08549>`_:

.. code-block:: tex

    @ARTICLE{2019MNRAS.490.2262E,
           author = {{Espinoza}, N{\'e}stor and {Kossakowski}, Diana and {Brahm}, Rafael},
            title = "{juliet: a versatile modelling tool for transiting and non-transiting exoplanetary systems}",
          journal = {\mnras},
         keywords = {methods: data analysis, methods: statistical, techniques: photometric, techniques: radial velocities, planets and satellites: fundamental parameters, planets and satellites: individual: K2-140b, K2-32b, c, d, Astrophysics - Earth and Planetary Astrophysics},
             year = "2019",
            month = "Dec",
           volume = {490},
           number = {2},
            pages = {2262-2283},
              doi = {10.1093/mnras/stz2688},
    archivePrefix = {arXiv},
           eprint = {1812.08549},
     primaryClass = {astro-ph.EP},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.2262E},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

Additional citations
---------------------

In addition to the citation above, and depending on the methods and samplers used in your research, please make sure to cite the appropiate sources:

* **If transit fits were performed**, cite ``batman``:

.. code-block:: tex  

    @ARTICLE{batman,
           author = {{Kreidberg}, Laura},
            title = "{batman: BAsic Transit Model cAlculatioN in Python}",
          journal = {Publications of the Astronomical Society of the Pacific},
         keywords = {Astrophysics - Earth and Planetary Astrophysics},
             year = 2015,
            month = Nov,
           volume = {127},
            pages = {1161},
              doi = {10.1086/683602},
    archivePrefix = {arXiv},
           eprint = {1507.08285},
     primaryClass = {astro-ph.EP},
           adsurl = {https://ui.adsabs.harvard.edu/\#abs/2015PASP..127.1161K},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

In addition, ``juliet`` allows to sample limb-darkening coefficients using the method outlined in `Kipping (2013) <https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract>`_. If using it, please cite:

.. code-block:: tex  

    @ARTICLE{2013MNRAS.435.2152K,
           author = {{Kipping}, David M.},
            title = "{Efficient, uninformative sampling of limb darkening coefficients for two-parameter laws}",
          journal = {\mnras},
         keywords = {methods: analytical, stars: atmospheres, Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Earth and Planetary Astrophysics},
             year = 2013,
            month = nov,
           volume = {435},
           number = {3},
            pages = {2152-2160},
              doi = {10.1093/mnras/stt1435},
    archivePrefix = {arXiv},
           eprint = {1308.0009},
     primaryClass = {astro-ph.SR},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

If using the uninformative sample for radius and impact parameters outlined in `Espinoza (2018) <https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..209E/exportcitation>`_, cite:

.. code-block:: tex

    @ARTICLE{2018RNAAS...2..209E,
           author = {{Espinoza}, N{\'e}stor},
            title = "{Efficient Joint Sampling of Impact Parameters and Transit Depths in Transiting Exoplanet Light Curves}",
          journal = {Research Notes of the American Astronomical Society},
         keywords = {Astrophysics - Earth and Planetary Astrophysics},
             year = 2018,
            month = nov,
           volume = {2},
           number = {4},
              eid = {209},
            pages = {209},
              doi = {10.3847/2515-5172/aaef38},
    archivePrefix = {arXiv},
           eprint = {1811.04859},
     primaryClass = {astro-ph.EP},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..209E},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

* **If radial-velocity fits were performed**, cite ``radvel``:

.. code-block:: tex 

    @ARTICLE{radvel,
       author = {{Fulton}, B.~J. and {Petigura}, E.~A. and {Blunt}, S. and {Sinukoff}, E.
            },
        title = "{RadVel: The Radial Velocity Modeling Toolkit}",
      journal = {\pasp},
    archivePrefix = "arXiv",
       eprint = {1801.01947},
     primaryClass = "astro-ph.IM",
         year = 2018,
        month = apr,
       volume = 130,
       number = 4,
        pages = {044504},
          doi = {10.1088/1538-3873/aaaaa8},
       adsurl = {http://adsabs.harvard.edu/abs/2018PASP..130d4504F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

* **If Gaussian Processes were used**, cite either ``george`` and/or ``celerite`` depending on the used kernel(s):

.. code-block:: tex 

     @article{george,
        author = {{Ambikasaran}, S. and {Foreman-Mackey}, D. and
                  {Greengard}, L. and {Hogg}, D.~W. and {O'Neil}, M.},
         title = "{Fast Direct Methods for Gaussian Processes}",
          year = 2014,
         month = mar,
           url = http://arxiv.org/abs/1403.6015
    }

.. code-block:: tex 

    @article{celerite,
        author = {{Foreman-Mackey}, D. and {Agol}, E. and {Angus}, R. and
                  {Ambikasaran}, S.},
         title = {Fast and scalable Gaussian process modeling
                  with applications to astronomical time series},
          year = {2017},
       journal = {AJ},
        volume = {154},
         pages = {220},
           doi = {10.3847/1538-3881/aa9332},
           url = {https://arxiv.org/abs/1703.09710}
    }

* **If MultiNest was used to perform the sampling**, cite ``MultiNest`` and ``PyMultiNest``:

.. code-block:: tex 

    @ARTICLE{MultiNest,
       author = {{Feroz}, F. and {Hobson}, M.~P. and {Bridges}, M.},
        title = "{MULTINEST: an efficient and robust Bayesian inference tool for cosmology and particle physics}",
      journal = {\mnras},
    archivePrefix = "arXiv",
       eprint = {0809.3437},
     keywords = {methods: data analysis , methods: statistical},
         year = 2009,
        month = oct,
       volume = 398,
        pages = {1601-1614},
          doi = {10.1111/j.1365-2966.2009.14548.x},
       adsurl = {http://adsabs.harvard.edu/abs/2009MNRAS.398.1601F},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @ARTICLE{PyMultiNest,
       author = {{Buchner}, J. and {Georgakakis}, A. and {Nandra}, K. and {Hsu}, L. and
            {Rangel}, C. and {Brightman}, M. and {Merloni}, A. and {Salvato}, M. and
            {Donley}, J. and {Kocevski}, D.},
        title = "{X-ray spectral modelling of the AGN obscuring region in the CDFS: Bayesian model selection and catalogue}",
      journal = {\aap},
    archivePrefix = "arXiv",
       eprint = {1402.0004},
     primaryClass = "astro-ph.HE",
     keywords = {accretion, accretion disks, methods: data analysis, methods: statistical, galaxies: nuclei, X-rays: galaxies, galaxies: high-redshift},
         year = 2014,
        month = apr,
       volume = 564,
          eid = {A125},
        pages = {A125},
          doi = {10.1051/0004-6361/201322971},
       adsurl = {http://adsabs.harvard.edu/abs/2014A%26A...564A.125B},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

* **If dynesty was used to perform the sampling**, cite ``dynesty``:

.. code-block:: tex

    @ARTICLE{2020MNRAS.493.3132S,
           author = {{Speagle}, Joshua S.},
            title = "{DYNESTY: a dynamic nested sampling package for estimating Bayesian posteriors and evidences}",
          journal = {\mnras},
         keywords = {methods: data analysis, methods: statistical, Astrophysics - Instrumentation and Methods for Astrophysics, Statistics - Computation},
             year = 2020,
            month = apr,
           volume = {493},
           number = {3},
            pages = {3132-3158},
              doi = {10.1093/mnras/staa278},
    archivePrefix = {arXiv},
           eprint = {1904.02180},
     primaryClass = {astro-ph.IM},
           adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.3132S},
          adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

* **If UltraNest was used to perform the sampling**, follow the instructions in the `UltraNest read-the-docs <https://johannesbuchner.github.io/UltraNest/issues.html#how-should-i-cite-ultranest>`_.
