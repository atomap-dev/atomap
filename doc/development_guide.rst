.. _development_guide:

This is a technical guide for people already familiar with scientific Python software development.
For an introduction to scientific Python software development, see :ref:`contribute`.

=================
Development guide
=================

Testing
-------

Atomap has a large number of unit tests, which are tested using ``pytest``:

.. code-block:: bash

    $ python3 -m pytest --doctest-modules atomap/

This also runs every docstring example as a unit test.

The documentation is tested by doing:

.. code-block:: bash

    $ python3 -m pytest --doctest-glob="*.rst" doc/


Both the unit tests and the doc tests can be accelerated by running the tests in parallel.
Use the `xdist <https://docs.pytest.org/en/3.0.0/xdist.html>`_ pytest package for this.
To run the tests using 5 parallel processes:

.. code-block:: bash

    $ python3 -m pytest -n 5 --doctest-modules atomap/
    $ python3 -m pytest -n 5 --doctest-glob="*.rst" doc/


Testing notebooks
*****************

The Jupyter notebooks in https://gitlab.com/atomap/atomap_demos is also tested using pytest
and ``nbval``.

.. code-block:: bash

      $ python3 -m pytest --nbval-lax introduction_to_atomap.ipynb


Note: for some reason the ``%matplotlib nbagg`` or ``%matplotlib qt`` causes the tests to fail,
the easiest way of avoiding this is skipping that specific notebook cell. This done
by adding ``nbval-skip`` to the tag for that cell.


Style checks
------------

In Atomap the PEP8 style guide is followed, and the `Black <https://black.readthedocs.io/en/stable/>`_
code formatter is used. To automatically format the code:

.. code-block:: bash

    $ python3 -m black atomap/


To only check the code, without changing it:

.. code-block:: bash

    $ python3 -m black atomap/ --diff --check


Generating the sphinx page
--------------------------
These documentation pages are written by using sphinx.
You generate the html site by:

.. code-block:: bash

    $ cd doc
    $ python3 -m sphinx -b html ./ _build/html/


Continuous integration
----------------------

The Continuous integration (CI) settings is contained in ``.gitlab-ci.yml``.
This runs all the above-mentioned tests, style checks and sphinx page generation on each branch.


Documentation from development branch
-------------------------------------

The most recent documentation generated from the development branch can be accessed `here <https://gitlab.com/atomap/atomap/builds/artifacts/master/file/pages_development/index.html?job=pages_development_branch>`_.
