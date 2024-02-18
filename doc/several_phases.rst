.. _several_phases:

==============
Several phases
==============

Many datasets contain different atomic structures, which should be processed separately in different sublattices.
For example precipitates in an Aluminium matrix.

.. code-block:: python

   >>> %matplotlib widget # doctest: +SKIP
   >>> import atomap.api as am
   >>> s = am.dummy_data.get_precipitate_signal()
   >>> s.plot()

.. image:: images/severalphases/precipitate_signal.png
    :scale: 50 %
    :align: center


Due to the different structures, we want to process the precipitate (in the centre) and the matrix separately.
The easiest way of doing this is using the :py:func:`~atomap.initial_position_finding.select_atoms_with_gui` function.
Firstly, find all the atom positions:

.. code-block:: python

   >>> atom_positions = am.get_atom_positions(s, 8)

Precipitate
===========

Then select the precipitate with :py:func:`~atomap.initial_position_finding.select_atoms_with_gui`:

.. code-block:: python

   >>> atoms_precipitate = am.select_atoms_with_gui(s, atom_positions)

.. image:: images/atomselectorgui/atom_selector_gui.gif
    :scale: 50 %
    :align: center

.. code-block:: python
   :hide:

   >>> verts = [[250, 100], [100, 250], [250, 400], [400, 250]]
   >>> atoms_precipitate = am.select_atoms_with_gui(s, atom_positions, verts=verts)


The region can also be selected non-interactively, by using the ``verts`` parameter.
See :ref:`atom_selector_gui` for an example of this.

We use this subset of atoms to create a sublattice

.. code-block:: python

   >>> sublattice_p = am.Sublattice(atoms_precipitate, s)
   >>> sublattice_p.plot()

.. image:: images/severalphases/precipitate_sublattice.png
    :scale: 50 %
    :align: center


Matrix
======

The atoms in the matrix is selected using the same function, but with ``invert_selection=True``.

.. code-block:: python

   >>> atoms_matrix = am.select_atoms_with_gui(s, atom_positions, invert_selection=True)

.. image:: images/atomselectorgui/atom_selector_invert_selection_gui.gif
    :scale: 50 %
    :align: center

.. code-block:: python
   :hide:

   >>> atoms_matrix = am.select_atoms_with_gui(s, atom_positions, verts=verts, invert_selection=True)


We use this subset of atoms to create a sublattice for the matrix

.. code-block:: python

   >>> sublattice_m = am.Sublattice(atoms_matrix, s, color='blue')
   >>> sublattice_m.plot()

.. image:: images/severalphases/matrix_sublattice.png
    :scale: 50 %
    :align: center


These two sublattices can then be added to an ``Atom_Lattice`` object.

.. code-block:: python

   >>> atom_lattice = am.Atom_Lattice(s, sublattice_list=[sublattice_p, sublattice_m])
   >>> atom_lattice.plot()

.. image:: images/severalphases/atom_lattice.png
    :scale: 50 %
    :align: center


Analysing the sublattices
=========================

Intensity
---------

Getting the intensity of the atomic columns can be done without doing any position refinement or other processing.

.. code-block:: python

   >>> i_points, i_record, p_record = atom_lattice.integrate_column_intensity()
   >>> i_record.plot()

.. image:: images/severalphases/atom_lattice_integrate.png
    :scale: 50 %
    :align: center

Note the higher intensity at the border of the image, which is due to the atoms at the edge of dataset not being identified as individual atoms.
So their intensity is added to the closest ones.
This effect can be reduced by using the ``max_radius`` parameter in :py:meth:`~atomap.atom_lattice.Atom_Lattice.integrate_column_intensity`, or by cropping the intensity output.

.. code-block:: python

   >>> i_record.isig[30:-30, 30:-30].plot()

.. image:: images/severalphases/atom_lattice_integrate_crop.png
    :scale: 50 %
    :align: center


Where ``isig`` is a method for cropping HyperSpy signals.


Distance between precipitate atoms
----------------------------------

See :ref:`getting_monolayer_distance` for more information.

Run position refinements for the precipitate sublattice, firstly by finding the atomic planes, and then refining the positions.

.. code-block:: python

   >>> sublattice_p.construct_zone_axes()
   >>> sublattice_p.refine_atom_positions_using_center_of_mass()
   >>> sublattice_p.refine_atom_positions_using_2d_gaussian()


Visualize this for the first zone axis:

.. code-block:: python

   >>> za0 = sublattice_p.zones_axis_average_distances[0]
   >>> s_mono0 = sublattice_p.get_monolayer_distance_map([za0])
   >>> s_mono0.plot()

.. image:: images/severalphases/precipitate_monolayer0.png
    :scale: 50 %
    :align: center
