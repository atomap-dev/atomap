from atomap.convert_ase import load_ase
from ase.cluster import Octahedron
import math


class TestASEImport:
    
    def test_simple(self):
        atoms = Octahedron('Ag', 5, cutoff=2)
        atomlattice = load_ase(atoms, (128,128), gaussian_blur=1)

        assert atomlattice.sublattice_list[0].atom_list[0].pixel_x == 0.0
        assert math.isclose(atomlattice.sublattice_list[0].atom_list[0].pixel_y,
                          29.09, abs_tol=0.01)
        assert atomlattice.sublattice_list[0].atom_list[0].element_info == {
            2.045: 'Ag', 6.135: 'Ag'}
