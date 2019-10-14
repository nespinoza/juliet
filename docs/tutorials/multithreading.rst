.. _quicktest:

Multithreading
===================

``juliet`` can be used in multiple cores in order to speed up the data fitting processes. If using ``MultiNest`` this is done via OpenMPI, whereas via ``dynesty`` this is done using internal ``python`` multi-threading capabilities. In what follows, we explain how to perform multiple core runs with ``juliet``.

Multithreading with MultiNest
-------------------------

In order to use the multi-threading capabilities with ``juliet``, you have to have OpenMPI in your computer. You can check if this is available in your system by opening a terminal and writing ``mpirun``. If this command prompts you to something similar to:

.. code-block:: bash

    --------------------------------------------------------------------------
    mpirun could not find anything to do.

    It is possible that you forgot to specify how many processes to run
    via the "-np" argument.
    --------------------------------------------------------------------------

Then that's it, you have OpenMPI. If not, installing it is simple. You just have to follow the instructions to compile OpenMPI [`here <https://www.open-mpi.org/faq/?category=building#easy-build>`_]. Once this is done, you have to install ``mpi4py``, which is easily done via ``pip``:

.. code-block:: bash
   
   pip install mpi4py

Once all this is done you are good to go! To run a juliet run on ``X`` number of cores, simply do:

.. code-block:: bash

   mpirun -np X python yourscript.py

Multithreading with dynesty
-----------------------------------------------

Applying multi-threading capabilities for ``dynesty`` is much simpler than for MultiNest. This can be 
automatically activated once a ``juliet.load`` object is made to fit the data --- simply define 
the number of threads you want to use and ``juliet`` will assume you need multi-threading capabilities. 
So, for example, to use ``juliet`` with 6 number of cores, in a session you would do:

.. code-block:: python

   # Load and fit dataset with juliet:
   dataset = juliet.load(priors=priors, t_lc = times, y_lc = fluxes, \
                      yerr_lc = fluxes_error, out_folder = 'hats46')

   results = dataset.fit(use_dynesty=True, dynesty_nthreads = 6)

