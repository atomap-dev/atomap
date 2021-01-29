import os
from hyperspy.io import load
from hyperspy.signals import Signal2D
import atomap.main as amm
import atomap.dummy_data as dd
from atomap.process_parameters import PerovskiteOxide110

my_path = os.path.dirname(__file__)


class TestMakeAtomLatticeFromImage:
    def setup_method(self):
        s_adf_filename = os.path.join(my_path, "datasets", "test_ADF_cropped.hdf5")
        self.s_adf = load(s_adf_filename)
        self.s_adf.change_dtype("uint16")
        self.pixel_separation = 19
        self.process_parameter = PerovskiteOxide110()

    def test_adf_image(self):
        s_adf = self.s_adf
        pixel_separation = self.pixel_separation
        process_parameter = self.process_parameter
        amm.make_atom_lattice_from_image(
            s_adf,
            process_parameter=process_parameter,
            pixel_separation=pixel_separation,
        )


class TestGetFilename:
    def setup_method(self):
        self.s = Signal2D([range(10), range(10)])

    def test_empty_metadata_and_tmp_parameters(self):
        s = self.s.deepcopy()
        filename = amm._get_signal_name(s)
        assert filename == "signal"

    def test_empty_metadata(self):
        s = self.s.deepcopy()
        s.__dict__["tmp_parameters"]["filename"] = "test2"
        filename = amm._get_signal_name(s)
        assert filename == "test2"

    def test_metadata(self):
        s = self.s.deepcopy()
        s.__dict__["tmp_parameters"]["filename"] = "test2"
        s.metadata.General.title = "test1"
        filename = amm._get_signal_name(s)
        assert filename == "test1"


class TestRunImageFiltering:
    def test_standard(self):
        s = dd.get_simple_cubic_signal()
        amm.run_image_filtering(s)

    def test_inverted_signal(self):
        s = dd.get_simple_cubic_signal()
        amm.run_image_filtering(s, invert_signal=True)
