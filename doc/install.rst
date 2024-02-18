.. _install:

==========
Installing
==========

.. _install_hyperspy_bundle:

HyperSpy bundle
---------------

The easiest way of installing Atomap in Windows, MacOS and Linux is by installing the HyperSpy bundle: `HyperSpy bundle <https://hyperspy.org/hyperspy-doc/current/user_guide/install.html#hyperspy-bundle>`_.
Currently, Atomap is not included in the bundle, but it will be included in the near future.
For now, Atomap must be installed manually by first opening the ``Hyperspy-bundle Prompt`` then running:

.. code-block:: bash

    $ conda install atomap -c conda-forge


If everything installed, continue to :ref:`starting Atomap <start_atomap>`.
If you got some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


.. _install_anaconda:

Anaconda Python distribution
----------------------------

First install the Anaconda python environment `Anaconda environment <https://www.anaconda.com/download>`_.
Then install Atomap via the ``Anaconda prompt`` (in Windows: Start menu - Anaconda3), this will open a command line prompt.
In this prompt run:

.. code-block:: bash

    $ conda install atomap -c conda-forge

If you also want to use the Atomap Jupyter Notebook tutorials, install Jupyter Lab and plotting library:

.. code-block:: bash

    $ conda install jupyterlab ipympl -c conda-forge


If everything installed, continue to :ref:`starting Atomap <start_atomap>`.
If you got some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


.. _install_pypi:

PyPI installation
-----------------

Atomap can also be installed via the Python respository PyPI.
This is not recommended for beginners, and requires that you already have a Python install.

.. code-block:: bash

    $ pip install atomap

If you got some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


Development version
-------------------

Grab the development version using the version control system git (see :ref:`contribute`):

.. code-block:: bash

    $ git clone https://gitlab.com/atomap/atomap.git

Then install it using pip:

.. code-block:: bash

    $ cd atomap
    $ pip3 install -e .
