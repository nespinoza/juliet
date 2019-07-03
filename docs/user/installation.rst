.. _installation:

Installation
===============

`juliet` can be easily installed using `pip <https://pip.pypa.io>`_:

.. code-block:: bash

    pip install juliet-package

The core of `juliet` is comprised of the transit (`batman <https://www.cfa.harvard.edu/~lkreidberg/batman/>`_, 
`starry <https://rodluger.github.io/starry/>`_), radial-velocity (`radvel <https://radvel.readthedocs.io/en/latest/>`_) 
and Gaussian Process (`george <https://george.readthedocs.io/en/latest/>`_, 
`celerite <https://celerite.readthedocs.io/en/stable/>`_) modelling tools, as well as 
of the Nested Sampling algorithms (`MultiNest` via `pymultinest <https://github.com/JohannesBuchner/PyMultiNest>`_, 
`dynesty <https://dynesty.readthedocs.io>`_) that it uses. However, by default the `juliet` installation will 
force `dynesty` as the main sampler to be installed, and only optionally install `pymultinest`. This is because 
the `pymultinest` installation can involve a couple of extra steps, which we really recommend following, as 
`pymultinest` might be faster for problems involving less than about 20 free parameters (see below).


.. _pymultinest_install:

Installing pymultinest
+++++++++++

The full instructions on how to install `pymultinest can be found in the project's documentation 
<http://johannesbuchner.github.io/PyMultiNest/install.html>`_. We repeat here the main steps. First, 
install it via `pip <https://pip.pypa.io>`_:

.. code-block:: bash

    pip install pymultinest

Then, you need to build and compile `MultiNest`. For this, do:

.. code-block:: bash

    git clone https://github.com/JohannesBuchner/MultiNest
    cd MultiNest/build
    cmake ..
    make

This will create a file `libmultinest.so` under `MultiNest/lib`, that is the one that will allow us 
to use `pymultinest`. Include that directory then in your `LD_LIBRARY_PATH` so you can use it from any 
directory in your system.

From Source
+++++++++++

The source code for `juliet` can be downloaded `from GitHub
<https://github.com/nespinoza/juliet>`_ by running

.. code-block:: bash

    git clone https://github.com/nespinoza/juliet.git


.. _python-deps:

**Dependencies**

For `juliet` to run you need a Python installation. After installation, you need to install:

1. `NumPy <http://www.numpy.org/>`_,
2. `SciPy <http://www.numpy.org/>`_,
3. `matplotlib <https://matplotlib.org/>`_, and
4. `seaborn <https://seaborn.pydata.org/>`.

The last two are optional, and are only needed for certain plotting functions within `juliet`.
