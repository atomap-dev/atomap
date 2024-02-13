.. _install:

==========
Installing
==========

.. _install_windows:

Installing in Windows
---------------------

Anaconda Python environment
***************************

Currently, the easiest way to install Atomap is using the Anaconda python environment `Anaconda environment <https://www.anaconda.com/download>`_,
Install HyperSpy, then Atomap via the ``Anaconda prompt`` (Start menu - Anaconda3), this will open a command line prompt.
In this prompt run:

.. code-block:: bash

    $ conda install atomap hyperspy-gui-traitsui hyperspy-gui-ipywidgets -c conda-forge

If everything installed, continue to :ref:`starting Atomap in Windows <start_atomap_windows>`.
If you got some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


Miniforge HyperSpy installer
****************************

Alternatively, the Miniforge HyperSpy bundle can be used.
Firstly download and install the `Miniforge HyperSpy bundle <https://github.com/hyperspy/hyperspy-bundle/releases>`_:

After installing the bundle, there should be a folder in the start menu called "HyperSpy Bundle", and this
folder should contain the "WinPython prompt". Start the "WinPython prompt". This will open a terminal window called
"WinPython prompt", in this window type and run:

.. code-block:: bash

    pip install atomap

If everything installed, continue to :ref:`starting Atomap in Windows <start_atomap_windows>`.
If you got some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


Installing in MacOS
-------------------

Install the Anaconda python environment: `Anaconda environment <https://www.anaconda.com/download>`_, and through the ``Anaconda prompt`` install HyperSpy and Atomap:

.. code-block:: bash

    $ conda install atomap hyperspy-gui-traitsui hyperspy-gui-ipywidgets -c conda-forge

If everything installed, continue to :ref:`starting Atomap in MacOS <start_atomap_macos>`.
If you got some kind of error, please report it as a New issue on the `Atomap GitLab <https://gitlab.com/atomap/atomap/issues>`_.


Installing in Linux
-------------------

Currently, the easiest way to install Atomap is using the Anaconda python environment `Anaconda environment <https://www.anaconda.com/download>`_,
Install HyperSpy, then Atomap via conda in the command line:

.. code-block:: bash

    $ conda install atomap hyperspy-gui-traitsui hyperspy-gui-ipywidgets -c conda-forge

If everything installed, continue to :ref:`starting Atomap in Linux <start_atomap_linux>`.
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
