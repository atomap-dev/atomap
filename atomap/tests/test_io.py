from os.path import join as pjoin
import tempfile
import pytest
import numpy as np
import atomap.atom_lattice as al
from atomap.sublattice import Sublattice
from atomap.io import load_atom_lattice_from_hdf5


class TestAtomLatticeInputOutput:
    def setup_method(self):
        image_data = np.arange(10000).reshape(100, 100)
        peaks0 = np.arange(20).reshape(10, 2)
        peaks1 = np.arange(26).reshape(13, 2)

        sublattice0 = Sublattice(atom_position_list=peaks0, image=image_data)
        sublattice1 = Sublattice(atom_position_list=peaks1, image=image_data)
        self.atom_lattice = al.Atom_Lattice()
        self.atom_lattice.sublattice_list.extend([sublattice0, sublattice1])
        self.atom_lattice.image0 = image_data
        self.tmpdir = tempfile.TemporaryDirectory()

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_save_load_atom_lattice_simple(self):
        save_path = pjoin(self.tmpdir.name, "test_atomic_lattice_save.hdf5")
        self.atom_lattice.save(filename=save_path, overwrite=True)
        load_atom_lattice_from_hdf5(save_path, construct_zone_axes=False)

    def test_save_load_atom_lattice_check_metadata_values(self):
        sublattice0 = self.atom_lattice.sublattice_list[0]
        sublattice1 = self.atom_lattice.sublattice_list[1]
        sublattice0.name = "test 0"
        sublattice1.name = "test 1"
        sublattice0.units = "pm"
        sublattice1.units = "pm"
        sublattice0.pixel_size = 0.01
        sublattice1.pixel_size = 0.01
        sublattice0._plot_color = "blue"
        sublattice1._plot_color = "green"
        assert len(sublattice0.atom_list) == 10
        assert len(sublattice1.atom_list) == 13

        save_path = pjoin(self.tmpdir.name, "test_atomic_lattice_save.hdf5")

        self.atom_lattice.save(filename=save_path, overwrite=True)
        atom_lattice_load = load_atom_lattice_from_hdf5(
            save_path, construct_zone_axes=False
        )
        sl0 = atom_lattice_load.sublattice_list[0]
        sl1 = atom_lattice_load.sublattice_list[1]

        assert len(sl0.atom_list) == 10
        assert len(sl1.atom_list) == 13
        assert sl0.name == "test 0"
        assert sl1.name == "test 1"
        assert sl0.units == "pm"
        assert sl1.units == "pm"
        assert sl0.pixel_size == 0.01
        assert sl1.pixel_size == 0.01
        assert sl0._plot_color == "blue"
        assert sl1._plot_color == "green"

    def test_save_load_atom_lattice_atom_values(self):
        image_data = np.arange(10000).reshape(100, 100)

        atom0_pos = np.random.random(size=(30, 2)) * 10
        atom0_sigma_x = np.random.random(size=30)
        atom0_sigma_y = np.random.random(size=30)
        atom0_rot = np.random.random(size=30)
        atom1_pos = np.random.random(size=(30, 2)) * 10
        atom1_sigma_x = np.random.random(size=30)
        atom1_sigma_y = np.random.random(size=30)
        atom1_rot = np.random.random(size=30)

        sublattice0 = Sublattice(atom_position_list=atom0_pos, image=image_data)
        sublattice1 = Sublattice(atom_position_list=atom1_pos, image=image_data)
        for i, atom in enumerate(sublattice0.atom_list):
            atom.sigma_x = atom0_sigma_x[i]
            atom.sigma_y = atom0_sigma_y[i]
            atom.rotation = atom0_rot[i]
        for i, atom in enumerate(sublattice1.atom_list):
            atom.sigma_x = atom1_sigma_x[i]
            atom.sigma_y = atom1_sigma_y[i]
            atom.rotation = atom1_rot[i]

        atom_lattice = al.Atom_Lattice()
        atom_lattice.sublattice_list.extend([sublattice0, sublattice1])
        atom_lattice.image0 = image_data

        save_path = pjoin(self.tmpdir.name, "atomic_lattice.hdf5")

        atom_lattice.save(filename=save_path, overwrite=True)
        atom_lattice_load = load_atom_lattice_from_hdf5(
            save_path, construct_zone_axes=False
        )
        sl0 = atom_lattice_load.sublattice_list[0]
        sl1 = atom_lattice_load.sublattice_list[1]

        assert (sl0.x_position == atom0_pos[:, 0]).all()
        assert (sl0.y_position == atom0_pos[:, 1]).all()
        assert (sl1.x_position == atom1_pos[:, 0]).all()
        assert (sl1.y_position == atom1_pos[:, 1]).all()
        assert (sl0.sigma_x == atom0_sigma_x).all()
        assert (sl0.sigma_y == atom0_sigma_y).all()
        assert (sl1.sigma_x == atom1_sigma_x).all()
        assert (sl1.sigma_y == atom1_sigma_y).all()
        assert (sl0.rotation == atom0_rot).all()
        assert (sl1.rotation == atom1_rot).all()

    def test_save_atom_lattice_already_exist(self):
        save_path = pjoin(self.tmpdir.name, "test_atomic_lattice_io.hdf5")
        self.atom_lattice.save(filename=save_path, overwrite=True)
        with pytest.raises(FileExistsError):
            self.atom_lattice.save(filename=save_path)

    def test_save_load_atom_lattice_type(self):
        save_path = pjoin(self.tmpdir.name, "test_atomic_lattice_save.hdf5")
        self.atom_lattice.save(filename=save_path, overwrite=True)
        atom_lattice_load = load_atom_lattice_from_hdf5(
            save_path, construct_zone_axes=False
        )
        al0_qualname = self.atom_lattice.__class__.__qualname__
        al1_qualname = atom_lattice_load.__class__.__qualname__
        assert al0_qualname == al1_qualname

    def test_save_load_element_info(self):
        save_path = pjoin(self.tmpdir.name, "test_atomic_lattice_save.hdf5")
        sublattice0 = self.atom_lattice.sublattice_list[0]
        sublattice0.set_element_info("C", [0.0, 0.5])
        self.atom_lattice.save(filename=save_path, overwrite=True)
        atom_lattice_load = load_atom_lattice_from_hdf5(
            save_path, construct_zone_axes=False
        )

        el_info = atom_lattice_load.sublattice_list[0].atom_list[0].element_info

        assert el_info[0.0] == "C"
        assert el_info[0.5] == "C"


class TestDumbbellLatticeType:
    def setup_method(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def teardown_method(self):
        self.tmpdir.cleanup()

    def test_load_simple(self):
        image_data = np.arange(10000).reshape(100, 100)
        peaks = np.arange(20).reshape(10, 2)
        sublattice = Sublattice(atom_position_list=peaks, image=image_data)
        dumbbell_lattice = al.Dumbbell_Lattice(
            image=image_data, sublattice_list=[sublattice, sublattice]
        )

        save_path = pjoin(self.tmpdir.name, "test_dumbbell_lattice_save.hdf5")
        dumbbell_lattice.save(filename=save_path, overwrite=True)
        dumbbell_lattice_load = load_atom_lattice_from_hdf5(
            save_path, construct_zone_axes=False
        )

        dl0_qualname = dumbbell_lattice.__class__.__qualname__
        dl1_qualname = dumbbell_lattice_load.__class__.__qualname__
        assert dl0_qualname == "Dumbbell_Lattice"
        assert dl0_qualname == dl1_qualname
