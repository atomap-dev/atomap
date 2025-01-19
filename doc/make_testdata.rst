.. _make_testdata:

===========================
Making datasets for testing
===========================

The functionality for generating specific test data is useful for

1. Testing processing tools on known standards
2. Finding parameter sensitivity for processing tools
3. Easily generate example datasets
4. Use in unit tests for Atomap

The :py:mod:`atomap.dummy_data` module contains several pre-generated structures:

.. code-block:: python

    >>> import atomap.api as am
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> s.plot()

.. image:: images/maketestdata/dumbbell.png
    :scale: 50 %
    :align: center

A full list of these pre-generated structures can be found in the API documentation for the dummy data module: :py:mod:`atomap.dummy_data`

In addition, your own structures can be generated using the :py:class:`atomap.testing_tools.MakeTestData` class.
This tutorial will show how to do this.

The class is initialized by specifying the size of the image, using ``image_x`` and ``image_y``.

.. code-block:: python

    >>> import atomap.testing_tools as tt
    >>> t1 =  tt.MakeTestData(image_x=20, image_y=20)
    >>> t1.add_atom(x=10, y=10)
    >>> t1.signal.plot()

.. image:: images/maketestdata/t1.png
    :scale: 50 %
    :align: center

To a MakeTestData object you can add single atoms, or lists of atoms.
Note that the ``x`` and ``y`` arrays needs to be flattened first.

.. code-block:: python

    >>> import numpy as np
    >>> t2 = tt.MakeTestData(200,200)
    >>> x, y = np.mgrid[0:200:10j, 0:200:10j]
    >>> x, y = x.flatten(), y.flatten()
    >>> t2.add_atom_list(x, y)
    >>> t2.signal.plot()

.. image:: images/maketestdata/t2.png
    :scale: 50 %
    :align: center

The key is to make lists of the x and y positions of the atoms in a sublattice.
In the above example, the *j* means that 10 positions will be distributed between 0 and 200.
Below, the exact separation between the positions is set.
You can add many sublattices, and for each sublattice you can set the properties of the Gaussian used to model the atoms.

.. code-block:: python

    >>> t3 = tt.MakeTestData(200, 200)
    >>> x, y = np.mgrid[0:200:20, 0:200:20]
    >>> x, y = x.flatten(), y.flatten()
    >>> t3.add_atom_list(x, y,sigma_x=2, sigma_y=1.5, amplitude=20, rotation=0.4)

    >>> x, y = np.mgrid[10:200:20, 10:200:20]
    >>> x, y = x.flatten(), y.flatten()
    >>> t3.add_atom_list(x, y,sigma_x=2, sigma_y=2, amplitude=40)
    >>> t3.add_image_noise(sigma=0.1)
    >>> t3.signal.plot()

.. image:: images/maketestdata/t3.png
    :scale: 50 %
    :align: center

In the last example, image noise is added.
The image noise is currently Gaussian distributed, and both the standard deviation and expectation value of the noise (mu) can be set.
By default, mu=0.

It is also possible to make sublattice and atom lattice objects:

.. code-block:: python

    >>> sublattice3 = t3.sublattice
    >>> atom_lattice3 = t3.atom_lattice


Large amount of atoms
=====================

Generating test signals with a large amount of atoms (2000+) can take a very long time.
However, sublattice objects can be generated quickly if the image is not needed.
This is done by using ``sublattice_generate_image=False``:

.. code-block:: python

    >>> testdata = tt.MakeTestData(1000, 1000, sublattice_generate_image=False)
    >>> x, y = np.mgrid[0:1000:150j, 0:1000:150j]
    >>> x, y = x.flatten(), y.flatten()
    >>> testdata.add_atom_list(x, y)
    >>> sublattice = testdata.sublattice

Note that this sublattice will not be useful for testing the position refinement and fitting functionality, as the image is all zeros.
However, it can be useful for testing fingerprinting and construction of atom planes.


Saving test signal as image file
================================

The image test data can be saved to an image file using ``save_as_image``.
This image file will have exactly the same amounts of pixels as defined in the ``MakeTestData`` object.
So for the following example ``test_data.png`` will be 200 time 200 pixels.
You can save in most image formats: png, tif, jpg and more.
Note that jpg is a lossy format, which might introduce compression artifacts.
If you need lossless images, use png or tif.

.. code-block:: python

    >>> import numpy as np
    >>> t4 = tt.MakeTestData(200, 200)
    >>> x, y = np.mgrid[0:200:10j, 0:200:10j]
    >>> x, y = x.flatten(), y.flatten()
    >>> t4.add_atom_list(x, y)
    >>> t4.save_as_image("test_data.png")
