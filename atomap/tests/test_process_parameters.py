from atomap import process_parameters


class TestProcessParameters:
    def setup_method(self):
        self.generic_structure = process_parameters.GenericStructure()

    def test_generic_structure(self):
        generic_structure = self.generic_structure
        sublattice0 = generic_structure.sublattice_list[0]
        sublattice1 = process_parameters.GenericSublattice()
        generic_structure.add_sublattice_config(sublattice1)
        assert sublattice1.sublattice_order == 1
        assert sublattice0.name != sublattice1.name
        assert sublattice0.color != sublattice1.color
