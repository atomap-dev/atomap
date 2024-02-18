.. _gui_functions:

=================
GUI functionality
=================

.. _atom_selector_gui:

Selecting atoms with GUI
========================

Many datasets contain different atomic structures, which should be processed separately in different sublattices.
One way of separating them is using :py:func:`~atomap.initial_position_finding.select_atoms_with_gui`.


.. code-block:: python

   >>> %matplotlib widget # doctest: +SKIP
   >>> import atomap.api as am
   >>> s = am.dummy_data.get_precipitate_signal()
   >>> atom_positions = am.get_atom_positions(s, 8)
   >>> atom_positions_selected = am.select_atoms_with_gui(s, atom_positions)

.. image:: images/atomselectorgui/atom_selector_gui.gif
    :scale: 50 %
    :align: center


The selection can be inverted by using the ``invert_selection=True``:

.. code-block:: python

   >>> atom_positions_selected = am.select_atoms_with_gui(s, atom_positions, invert_selection=True)

.. image:: images/atomselectorgui/atom_selector_invert_selection_gui.gif
    :scale: 50 %
    :align: center


The function can also be used non-interactively via the ``verts`` parameter, which is useful when writing processing scripts:

.. code-block:: python

   >>> verts = [[250, 100], [100, 250], [250, 400], [400, 250], [250, 100], [-10, 80]]
   >>> atom_positions_selected = am.select_atoms_with_gui(s, atom_positions, verts=verts)


For an example of how to use this function to analyse a precipitate/matrix system, see :ref:`several_phases`.


.. _atom_adder_gui:

Adding atoms using GUI
======================

For most cases the majority of the atoms is found with automatic peak finding using :py:func:`~atomap.atom_finding_refining.get_feature_separation` and :py:func:`~atomap.atom_finding_refining.get_atom_positions`.
However, for some datasets there might be either missing or extra atoms.
These can be added or removed using :py:func:`~atomap.initial_position_finding.add_atoms_with_gui`.
This function opens up a window showing the datasets, where atoms can be added or removed by clicking on them with the mouse pointer.


.. code-block:: python

   >>> %matplotlib widget # doctest: +SKIP
   >>> s = am.dummy_data.get_distorted_cubic_signal()
   >>> atom_positions = am.get_atom_positions(s, 25)
   >>> atom_positions_new = am.add_atoms_with_gui(s, atom_positions)

.. image:: images/atomadderremovergui/atoms_add_remove_gui.gif
    :scale: 50 %
    :align: center


After having added or removed the atoms, ``atom_positions_new`` is used to make a sublattice object:

.. code-block:: python

   >>> sublattice = am.Sublattice(atom_positions_new, s)


:py:func:`~atomap.initial_position_finding.add_atoms_with_gui` can also be used without any initial atoms:


.. code-block:: python

   >>> atom_positions = am.add_atoms_with_gui(s)


If the atoms in the dataset are too close together, ``distance_threshold`` is used to decrease the distance for removing an atom.


.. code-block:: python

   >>> atom_positions = am.add_atoms_with_gui(s, distance_threshold=2)


If some of the atoms have much lower intensity than the others, the image can be shown in a log plot with the parameter ``norm='log'``.

.. code-block:: python

   >>> atom_positions = am.add_atoms_with_gui(s, norm='log')


.. _toggle_atom_refine:

Toggle atom refine
==================

To disable position refining or fitting of atoms in a sublattice, use :py:meth:`~atomap.sublattice.Sublattice.toggle_atom_refine_position_with_gui`:

.. code-block:: python

   >>> %matplotlib qt # doctest: +SKIP
   >>> sublattice = am.dummy_data.get_distorted_cubic_sublattice()
   >>> sublattice.toggle_atom_refine_position_with_gui()

Use the left mouse button to toggle refinement of the atom positions.
Green: refinement.
Red: not refinement.

.. image:: images/togglerefineposition/toggle_refine_position.gif
    :scale: 50 %
    :align: center

This can also be set directly through the `refine_position` property in the `Atom_Position` objects.

.. code-block:: python

   >>> sublattice.atom_list[5].refine_position = False
