.. _working_with_atomic_models:

==========================
Working with atomic models
==========================

Atomap provides functions to both export data to atomic models and to import data from atomic models.
This functionality is realised through interaction with the Atoms object of the `Atomic Simulation Environment <https://wiki.fysik.dtu.dk/ase/>`_ (ASE) package.

Exporting to atomic models
==========================

:py:class:`atomap.atom_lattice.Atom_Lattice` objects can be converted to an ASE Atoms object using the :py:meth:`~atomap.atom_lattice.Atom_Lattice.convert_to_ase` function.
Before conversion, all :py:class:`atomap.atom_position.Atom_Position` objects must have their :py:attr:`~atomap.atom_position.Atom_Position.element_info` set.
The easiest way to set :py:attr:`~atomap.atom_position.Atom_Position.element_info` for each atom position is to set the same values for all atoms in a :py:class:`~atomap.sublattice.Sublattice`.
For example:

.. code-block:: python

    >>> import atomap.api as am
    >>> atom_lattice = am.dummy_data.get_fantasite_atom_lattice()
    >>> atom_lattice.sublattice_list[0].set_element_info('Sr', [0.5])
    >>> atom_lattice.sublattice_list[1].set_element_info(['Ti', 'O'], [0.0, 1.0])
    >>> atoms = atom_lattice.convert_to_ase()

The converted Atoms object can now be saved as an atomic structure file (xyz, cif, etc...), can be visualized in 3D and can be input to atomistic simulations.

Visualization of atomic structure in 3D:

.. code-block:: python

    >>> from ase.visualize import view
    >>> view(atoms) # doctest: +SKIP

Note: In order to accurately convert an atomic structure, the :py:attr:`~atomap.sublattice.Sublattice.pixel_size` attribute must be set correctly for each :py:class:`~atomap.sublattice.Sublattice`.

Importing atomic models
=======================

It is also possible to import an atomic model from ASE, to create an :py:class:`~atomap.atom_lattice.Atom_Lattice` object.
To do this, use the :py:func:`~atomap.convert_ase.load_ase` function.
For example, to import a nanoparticle example dataset from ASE:

.. code-block:: python

    >>> import atomap.api as am
    >>> from ase.cluster import Octahedron
    >>> atoms = Octahedron('Ag', 10, cutoff=2)
    >>> atomlattice = am.ase_to_atom_lattice(atoms, (128, 128), gaussian_blur=1)
