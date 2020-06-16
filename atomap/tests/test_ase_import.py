from atomap.convert_ase import load_ase
from ase.cluster import Octahedron
from unittest import assertAlmostEqual


class TestASEImport:
    
    def test_simple(self):
        atoms = Octahedron('Ag', 5, cutoff=2)
        atomlattice = load_ase(atoms, (128,128), gaussian_blur=1)

        assert atomlattice.sublattice_list[0].atom_list[0].pixel_x == 0.0
        assertAlmostEqual(atomlattice.sublattice_list[0].atom_list[0].pixel_y,
                          29.09)
        assert atomlattice.sublattice_list[0].atom_list[0].element_info == {
            2.045: 'Ag', 6.135: 'Ag'}
