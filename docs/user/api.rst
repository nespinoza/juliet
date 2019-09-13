.. _api:

API
====
.. module:: juliet

The core classes within ``juliet`` are the ``load`` and ``fit`` classes. When creating a ``juliet.load`` object, the returned object will be able to call a 
``fit`` function which in turn returns a ``juliet.fit`` object, which saves all the information about the fit (results statistics, posteriors, model evaluations, 
etc.) --- these classes are explained in detail below:

.. autoclass:: juliet.load
   :members:

.. autoclass:: juliet.fit
   :members:

The returned ``fit`` object, in turn, also has other objects inherted in it. In particular, if ``results`` is a ``juliet.fit`` object, ``results.lc`` and ``results.rv`` 
are ``juliet.model`` objects that host all the details about the dataset being modelled. This follows the model definition outlined in Section 2 of the 
`juliet paper <https://arxiv.org/abs/1812.08549>`_:

.. autoclass:: juliet.model
   :members:

Finally, the ``juliet.load`` object also contains a dictionary (``juliet.load.lc_options`` for lightcurves and ``juliet.load.rv_options`` for radial-velocities) 
which holds, if a gaussian-process is being used to model the noise, a ``juliet.gaussian_process`` object. This class handles everything related to the gaussian-processes, 
from model and parameter names/values, to log-likelihood evaluations. This class is defined below:

.. autoclass:: juliet.gaussian_process
   :members:
