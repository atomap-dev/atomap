.. _start_atomap:

============
Start Atomap
============

The first step is starting an interactive JupyterLab environment.

This depends on the installation method:

* If the HyperSpy bundle was installed, go to *HyperSpy-bundle* in the start-menu, click *Hyperspy-bundle prompt*. This will open a command line window, here run ``jupyter lab``.
* If Anaconda was used, go to *Anaconda3* in the start menu, click *Anaconda prompt*. This will open a command line window, here run ``jupyter lab``.
* If you're using MacOS or Linux, run ``jupyter lab`` in your Anaconda environment or HyperSpy-bundle environment.

This will open a browser window (or a new browser tab).
Click ``Python 3 (ipykernel)``, which will start and open an empty Jupyter Notebook.

In the first cell, run the following commands (paste them, and press Shift + Enter).
If you are unfamiliar with the Jupyter Notebook interface, `see the interactive JupyterLab guide <https://jupyter.org/try-jupyter/lab/?path=notebooks%2FIntro.ipynb>`_.

.. code-block:: python

    %matplotlib widget
    import atomap.api as am

If this works, continue to the :ref:`tutorials`.
If you get some kind of error, please report it as a New issue on the `Atomap GitHub <https://github.com/atomap-dev/atomap/issues>`_.


.. _tutorials:

Tutorials
---------

To get you started on using Atomap there are tutorials available.
The first tutorial :ref:`finding_atom_lattices` aims at showing how atom positions are found, while :ref:`analysing_atom_lattices` shows how this information can be visualized.

The `>>>` used in the tutorials and documentation means the comment should be typed inside some kind of Python prompt, and can be copy-pasted directly into the *Jupyter Notebooks*.


Atomap demos
^^^^^^^^^^^^

In addition to the guides on this webpage, another good resource is the `Atomap demos <https://github.com/atomap-dev/atomap-demos/tree/release>`_, which are pre-filled Jupyter Notebooks showing various aspects of Atomap's functionality.
For beginners, the `Introduction to Atomap notebook <https://github.com/atomap-dev/atomap-demos/blob/release/introduction_to_atomap.ipynb>`_ is a good place to start.
