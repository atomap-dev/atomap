from atomap.convert_ase import ase_to_atom_lattice
from ase.cluster import Octahedron
import math


class TestASEImport:
    def test_simple(self):
        atoms = Octahedron("Ag", 5, cutoff=2)
        atomlattice = ase_to_atom_lattice(atoms, (128, 128), gaussian_blur=1)

        assert len(atomlattice.sublattice_list[0].atom_list) == 25
