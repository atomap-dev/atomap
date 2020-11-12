color_name_list = ["red", "blue", "green", "purple", "cyan", "yellow"]


class SublatticeParameterBase:
    def __init__(self):
        self.color = "red"
        self.name = "Base Sublattice"
        self.sublattice_order = None

    def __repr__(self):
        return "<%s, %s>" % (self.__class__.__name__, self.name)


class GenericSublattice(SublatticeParameterBase):

    """Process parameters for a generic sublattice

    Attributes
    ----------
    color : 'red' , color of the markers indicating atom positions
    image_type : 0
        The image will not be inverted.
    name : 'S0' , name of the sublattice
    sublattice_order : 0
        The sublattice will get the order 0.
        Higher orders can be used for less intense sublattices in
        images with multiple sublattices.
    zone_axis_list : list
        Can have up to 11 zone axis with name from 0-10.
    refinement_config : dict
        Dict with configuration settings for the refinement of atom positions.
        1st refinement : center-of-mass on image modified with background
        removal, PCA noise filtering and normalization.
        2nd refinement : Center of mass on the original image.
        3rd refinement : Fitting 2D Gaussians to the original image.
    neighbor_distance : 0.35
        Mask radius for fitting, set to 0.35 of the distance to
        nearest neighbor.
    atom_subtract_config : list
        Configuration for how to subtract higher order sublattices
        from the image. (Not really used in this structure, but included
        as an example.)

    Examples
    --------
    >>> import atomap.process_parameters as pp
    >>> generic_sublattice = pp.GenericSublattice()

    """

    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.color = "red"
        self.image_type = 0
        self.name = "S0"
        self.sublattice_order = 0
        self.zone_axis_list = [
            {"number": 0, "name": "0"},
            {"number": 1, "name": "1"},
            {"number": 2, "name": "2"},
            {"number": 3, "name": "3"},
            {"number": 4, "name": "4"},
            {"number": 5, "name": "5"},
            {"number": 6, "name": "6"},
            {"number": 7, "name": "7"},
            {"number": 8, "name": "8"},
            {"number": 9, "name": "9"},
            {"number": 10, "name": "10"},
        ]
        self.refinement_config = {
            "config": [
                ["image_data_modified", 1, "center_of_mass"],
                ["image_data", 1, "center_of_mass"],
                ["image_data", 1, "gaussian"],
            ],
            "neighbor_distance": 0.35,
        }
        self.atom_subtract_config = [
            {
                "sublattice": "S0",
                "neighbor_distance": 0.35,
            },
        ]


class PerovskiteOxide110SublatticeACation(SublatticeParameterBase):
    """Process parameters for the most intense atoms (typically the A-cations)
    in a Perovskite Oxide structure projected along the (110) direction.

    Attributes
    ----------
    color : 'blue' , color of the markers indicating atom positions
    image_type : 0
        The image will not be inverted for finding atom positions.
    name : 'A-cation' , name of the sublattice
    sublattice_order : 0
        The most intense sublattice gets order 0.
    zone_axis_list : list
         A list of numbers and names for the zone axes in this projection.
    refinement_config : dict
        Dict with configuration settings for the refinement of atom positions.
        Two refinements by fitting 2D Gaussians to the original image.
    neighbor_distance : 0.35
        Mask radius for fitting set to 0.35 of nearest neighbor.

    """

    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.name = "A-cation"
        self.color = "blue"
        self.image_type = 0
        self.zone_axis_list = [
            {"number": 0, "name": "110"},
            {"number": 1, "name": "100"},
            {"number": 2, "name": "11-2"},
            {"number": 3, "name": "112"},
            {"number": 4, "name": "111"},
            {"number": 5, "name": "11-1"},
        ]
        self.sublattice_order = 0
        self.refinement_config = {
            "config": [
                ["image_data", 2, "gaussian"],
            ],
            "neighbor_distance": 0.35,
        }


class PerovskiteOxide110SublatticeBCation(SublatticeParameterBase):
    """Process parameters for the second most intense atoms
    (typically the B-cations) in a Perovskite Oxide structure projected along
    the (110) direction.

    Attributes
    ----------
    color : 'green' , color of the markers indicating atom positions
    image_type : 0
        The image will not be inverted.
    name : 'B-cation' , name of the sublattice
    sublattice_order : 1
        The sublattice will get the order 1.
        Higher orders can be used for less intense sublattices in
        images with multiple sublattices. Lower order is for more intense
        sublattices.
    zone_axis_list : list
         A list of numbers and names for the zone axes in this projection.
    refinement_config : dict
        Dict with configuration settings for the refinement of atom positions.
        1st refinement : center-of-mass on the original image.
        2nd refinement : Fitting 2D Gaussians to the original image.
    neighbor_distance : 0.25
        Mask radius for fitting set to 0.25 of nearest neighbor.
    atom_subtract_config : list
        Configuration for how to subtract higher order sublattices
        from the image. Subtracts a sublattice with name 'A-cation' from the
        original image.

    """

    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.name = "B-cation"
        self.color = "green"
        self.image_type = 0
        self.zone_axis_list = [
            {"number": 0, "name": "110"},
            {"number": 1, "name": "100"},
            {"number": 2, "name": "11-2"},
            {"number": 3, "name": "112"},
            {"number": 4, "name": "111"},
            {"number": 5, "name": "11-1"},
        ]
        self.sublattice_order = 1
        self.sublattice_position_sublattice = "A-cation"
        self.sublattice_position_zoneaxis = "100"
        self.refinement_config = {
            "config": [
                ["image_data", 1, "center_of_mass"],
                ["image_data", 1, "gaussian"],
            ],
            "neighbor_distance": 0.25,
        }
        self.atom_subtract_config = [
            {
                "sublattice": "A-cation",
                "neighbor_distance": 0.35,
            },
        ]


class PerovskiteOxide110SublatticeOxygen(SublatticeParameterBase):
    """Process parameters for the least intense atoms
    (typically the Oxygen) in a Perovskite Oxide structure projected along
    the (110) direction.

    Attributes
    ----------
    color : 'red' , color of the markers indicating atom positions
    image_type : 1
        The image will be inverted. Oxygen can be visible in ABF images.
        The image is inverted such that the atoms are bright dots on a dark
        background.
    name : 'Oxygen' , name of the sublattice
    sublattice_order : 2
        The sublattice will get the order 2, being the third most intense
        sublattice.
    zone_axis_list : list
         A list of numbers and names for the zone axes in this projection.
    refinement_config : dict
        Dict with configuration settings for the refinement of atom positions.
        1st refinement : center-of-mass on the original image.
        2nd refinement : Fitting 2D Gaussians to the original image.
    neighbor_distance : 0.25
        Mask radius for fitting set to 0.25 of nearest neighbor.
    atom_subtract_config : list
        Configuration for how to subtract higher order sublattices
        from the image. Subtracts first a sublattice with name 'A-cation'
        from the inverted image, then a sublattice with name 'B-cation'.

    """

    def __init__(self):
        SublatticeParameterBase.__init__(self)
        self.name = "Oxygen"
        self.color = "red"
        self.image_type = 1
        self.zone_axis_list = [
            {"number": 0, "name": "110"},
            {"number": 1, "name": "100"},
            {"number": 2, "name": "11-2"},
            {"number": 3, "name": "112"},
            {"number": 4, "name": "111"},
            {"number": 5, "name": "11-1"},
        ]
        self.sublattice_order = 2
        self.sublattice_position_sublattice = "B-cation"
        self.sublattice_position_zoneaxis = "110"
        self.refinement_config = {
            "config": [
                ["image_data", 1, "center_of_mass"],
                ["image_data", 1, "gaussian"],
            ],
            "neighbor_distance": 0.25,
        }
        self.atom_subtract_config = [
            {
                "sublattice": "A-cation",
                "neighbor_distance": 0.35,
            },
            {
                "sublattice": "B-cation",
                "neighbor_distance": 0.30,
            },
        ]


class ModelParametersBase:
    def __init__(self):
        self.peak_separation = None
        self.name = None
        self.sublattice_list = []

    def __repr__(self):
        return "<%s, %s>" % (
            self.__class__.__name__,
            self.name,
        )

    def get_sublattice_from_order(self, order_number):
        for sublattice in self.sublattice_list:
            if order_number == sublattice.sublattice_order:
                return sublattice
        return False

    @property
    def number_of_sublattices(self):
        return len(self.sublattice_list)

    def add_sublattice_config(self, sublattice_config_object):
        name_list = []
        color_list = []
        sublattice_order_list = []
        for sublattice in self.sublattice_list:
            color_list.append(sublattice.color)
            sublattice_order_list.append(sublattice.sublattice_order)
            name_list.append(sublattice.name)
        sublattice_config_object.sublattice_order = max(sublattice_order_list) + 1

        if sublattice_config_object.color in color_list:
            for color in color_name_list:
                if not (color in color_list):
                    sublattice_config_object.color = color

        if sublattice_config_object.name in name_list:
            for i in range(20):
                name = "Sublattice " + str(i)
                if not (name in name_list):
                    sublattice_config_object.name = name
                    break

        self.sublattice_list.append(sublattice_config_object)


class GenericStructure(ModelParametersBase):

    """A Generic structure with one sublattice, the GenericSublattice.

    Parameters
    ----------
    peak_separation : None
        No peak separation is set, pixel_separation must be given separately
        in order to find initial atom positions.
    name : 'A structure'
    sublattice_list : list of Sublattices
        Contains one GenericSublattice

    """

    def __init__(self):
        ModelParametersBase.__init__(self)
        self.peak_separation = None
        self.name = "A structure"

        self.sublattice_list = [
            GenericSublattice(),
        ]


class PerovskiteOxide110(ModelParametersBase):

    """The Perovskite Oxide structure in the 110 projection.

    Parameters
    ----------
    name : 'Perovskite 110'
    peak_separation : 0.127 , distance in nm
        Approximately half the distance between the most intense atoms
        in the structure, used to get initial atom position by
        peak finding.
    sublattice_list : list of Sublattices
        Contains 3 sublattices, for A, B and O:
        PerovskiteOxide110SublatticeACation,
        PerovskiteOxide110SublatticeBCation,
        PerovskiteOxide110SublatticeOxygen

    """

    def __init__(self):
        ModelParametersBase.__init__(self)
        self.name = "Perovskite 110"
        self.peak_separation = 0.127

        self.sublattice_list = [
            PerovskiteOxide110SublatticeACation(),
            PerovskiteOxide110SublatticeBCation(),
            PerovskiteOxide110SublatticeOxygen(),
        ]


class SrTiO3_110(PerovskiteOxide110):
    def __init__(self):
        PerovskiteOxide110.__init__(self)
        self.sublattice_names = "Sr", "Ti", "O"
        Ti_sublattice_position = {"sublattice": "Sr", "zoneaxis": "100"}
        O_sublattice_position = {"sublattice": "Ti", "zoneaxis": "110"}
        self.sublattice_position = [Ti_sublattice_position, O_sublattice_position]
