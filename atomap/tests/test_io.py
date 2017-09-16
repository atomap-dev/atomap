import matplotlib
matplotlib.use('Agg')
import os
import tempfile
import unittest
import numpy as np
from atomap.atom_lattice import Atom_Lattice
from atomap.sublattice import Sublattice
from atomap.io import load_atom_lattice_from_hdf5

class test_atom_lattice_input_output(unittest.TestCase):

    def setUp(self):
        image_data = np.arange(10000).reshape(100, 100)
        peaks0 = np.arange(20).reshape(10, 2)
        peaks1 = np.arange(26).reshape(13, 2)

        sublattice0 = Sublattice(
                atom_position_list=peaks0,
                image=image_data)
        sublattice1 = Sublattice(
                atom_position_list=peaks1,
                image=image_data)
        self.atom_lattice = Atom_Lattice()
        self.atom_lattice.sublattice_list.extend([sublattice0, sublattice1])
        self.atom_lattice.image0 = image_data

    def test_save_load_atom_lattice_simple(self):
        tmpdir = tempfile.TemporaryDirectory()
        save_path = os.path.join(tmpdir.name, "test_atomic_lattice_save.hdf5")
        self.atom_lattice.save(
                filename=save_path, overwrite=True)
        load_atom_lattice_from_hdf5(save_path, construct_zone_axes=False)
        tmpdir.cleanup()

    def test_save_load_atom_lattice_check_metadata_values(self):
        sublattice0 = self.atom_lattice.sublattice_list[0]
        sublattice1 = self.atom_lattice.sublattice_list[1]
        sublattice0.name = "test 0"
        sublattice1.name = "test 1"
        sublattice0._plot_color = "blue"
        sublattice1._plot_color = "green"
        self.assertEqual(len(sublattice0.atom_list), 10)
        self.assertEqual(len(sublattice1.atom_list), 13)

        tmpdir = tempfile.TemporaryDirectory()
        save_path = os.path.join(tmpdir.name, "test_atomic_lattice_save.hdf5")

        self.atom_lattice.save(
                filename=save_path, overwrite=True)
        atom_lattice_load = load_atom_lattice_from_hdf5(
                save_path, construct_zone_axes=False)
        sl0 = atom_lattice_load.sublattice_list[0]
        sl1 = atom_lattice_load.sublattice_list[1]

        self.assertEqual(len(sl0.atom_list), 10)
        self.assertEqual(len(sl1.atom_list), 13)
        self.assertEqual(sl0.name, "test 0")
        self.assertEqual(sl1.name, "test 1")
        self.assertEqual(sl0._plot_color, "blue")
        self.assertEqual(sl1._plot_color, "green")

        tmpdir.cleanup()

    def test_save_load_atom_lattice_atom_values(self):
        image_data = np.arange(10000).reshape(100, 100)

        atom0_pos = np.random.random(size=(30, 2))*10
        atom0_sigma_x = np.random.random(size=30)
        atom0_sigma_y = np.random.random(size=30)
        atom0_rot = np.random.random(size=30)
        atom1_pos = np.random.random(size=(30, 2))*10
        atom1_sigma_x = np.random.random(size=30)
        atom1_sigma_y = np.random.random(size=30)
        atom1_rot = np.random.random(size=30)

        sublattice0 = Sublattice(
                atom_position_list=atom0_pos,
                image=image_data)
        sublattice1 = Sublattice(
                atom_position_list=atom1_pos,
                image=image_data)
        for i, atom in enumerate(sublattice0.atom_list):
            atom.sigma_x = atom0_sigma_x[i]
            atom.sigma_y = atom0_sigma_y[i]
            atom.rotation = atom0_rot[i]
        for i, atom in enumerate(sublattice1.atom_list):
            atom.sigma_x = atom1_sigma_x[i]
            atom.sigma_y = atom1_sigma_y[i]
            atom.rotation = atom1_rot[i]

        atom_lattice = Atom_Lattice()
        atom_lattice.sublattice_list.extend([sublattice0, sublattice1])
        atom_lattice.image0 = image_data

        tmpdir = tempfile.TemporaryDirectory()
        save_path = os.path.join(tmpdir.name, "atomic_lattice.hdf5")

        atom_lattice.save(filename=save_path, overwrite=True)
        atom_lattice_load = load_atom_lattice_from_hdf5(
                save_path, construct_zone_axes=False)
        sl0 = atom_lattice_load.sublattice_list[0]
        sl1 = atom_lattice_load.sublattice_list[1]

        self.assertTrue((sl0.x_position==atom0_pos[:,0]).all())
        self.assertTrue((sl0.y_position==atom0_pos[:,1]).all())
        self.assertTrue((sl1.x_position==atom1_pos[:,0]).all())
        self.assertTrue((sl1.y_position==atom1_pos[:,1]).all())
        self.assertTrue((sl0.sigma_x==atom0_sigma_x).all())
        self.assertTrue((sl0.sigma_y==atom0_sigma_y).all())
        self.assertTrue((sl1.sigma_x==atom1_sigma_x).all())
        self.assertTrue((sl1.sigma_y==atom1_sigma_y).all())
        self.assertTrue((sl0.rotation==atom0_rotation).all())
        self.assertTrue((sl1.rotation==atom1_rotation).all())


    @unittest.expectedFailure
    def test_save_atom_lattice_already_exist(self):
        tmpdir = tempfile.TemporaryDirectory()
        save_path = os.path.join(tmpdir.name, "test_atomic_lattice_io.hdf5")
        self.atom_lattice.save(
                filename=save_path, overwrite=True)
        self.atom_lattice.save(
                filename=save_path)
        tmpdir.cleanup()


