.. _installation:

Installation
===============

.. _pip_install:

Installing via pip
+++++++++++

``juliet`` can be easily installed using `pip <https://pip.pypa.io>`_:

.. code-block:: bash

    pip install juliet

The core of ``juliet`` is comprised of the transit (`batman <https://www.cfa.harvard.edu/~lkreidberg/batman/>`_, 
`starry <https://rodluger.github.io/starry/>`_), radial-velocity (`radvel <https://radvel.readthedocs.io/en/latest/>`_) 
and Gaussian Process (`george <https://george.readthedocs.io/en/latest/>`_, 
`celerite <https://celerite.readthedocs.io/en/stable/>`_) modelling tools, as well as 
of the Nested Sampling algorithms (`MultiNest` via `pymultinest <https://github.com/JohannesBuchner/PyMultiNest>`_, 
`dynesty <https://dynesty.readthedocs.io>`_) that it uses. However, **by default the ``juliet`` installation will 
force `dynesty` as the main sampler to be installed, and will not install `pymultinest`**. This is because 
the ``pymultinest`` installation can involve a couple of extra steps, which we really recommend following, as 
``pymultinest`` might be faster for problems involving less than about 20 free parameters (see below).


.. _source_install:

Installing from source
+++++++++++

The source code for ``juliet`` can be downloaded `from GitHub
<https://github.com/nespinoza/juliet>`_ by running

.. code-block:: bash

    git clone https://github.com/nespinoza/juliet.git

Once cloned, simply enter the ``juliet`` folder and do

.. code-block:: bash

    python setup.py install

To install the latest version of the code.

.. _pymultinest_install:

Installing pymultinest
+++++++++++

As described above, we really recommend installyng ``pymultinest``. The full instructions on how to install 
`pymultinest can be found in the project's documentation <http://johannesbuchner.github.io/PyMultiNest/install.html>`_. 
We repeat here the main steps. First, install it via `pip <https://pip.pypa.io>`_:

.. code-block:: bash

    pip install pymultinest

Then, you need to build and compile `MultiNest`. For this, do: 

.. code-block:: bash

    git clone https://github.com/JohannesBuchner/MultiNest
    cd MultiNest/build
    cmake ..
    make

This will create a file ``libmultinest.so`` or ``libmultinest.dylib`` under ``MultiNest/lib``: that is the one that will allow us  
to use ``pymultinest``. To not move that file around in your system, you can include the ``MultiNest/lib`` folder in your 
``LD_LIBRARY_PATH`` (e.g., in your ``~/.bash_profile`` or ``~/.bashrc`` file). In my case, the library is under ``/Users/nespinoza/github/MultiNest/lib``, so I added the following line to my ``~/.bash_profile`` file:

.. code-block:: bash

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/Users/nespinoza/github/MultiNest/lib

.. _python-deps:

**Dependencies**

The above installation instructuins for ``juliet`` assume you have a Python installation. ``juliet``, in turn, 
depends on the following libraries/packages, all of which will be installed automatically if you follow the instructions 
above:

1. `NumPy <http://www.numpy.org/>`_,
2. `SciPy <http://www.numpy.org/>`_,
3. `batman <https://www.cfa.harvard.edu/~lkreidberg/batman/>`_,
4. `radvel <https://radvel.readthedocs.io/en/latest/>`_,
5. `george <https://george.readthedocs.io/en/latest/>`_,
6. `celerite <https://celerite.readthedocs.io/en/stable/>`_,
7. `dynesty <https://dynesty.readthedocs.io>`_,
8. `pymultinest <https://github.com/JohannesBuchner/PyMultiNest>`_ (optional),
9. `matplotlib <https://matplotlib.org/>`_ (optional), and
10. `seaborn <https://seaborn.pydata.org/>`_ (optional).

The last are only needed for certain plotting functions within ``juliet``. The ``pymultinest`` installation is optional, but highly recommended. 
