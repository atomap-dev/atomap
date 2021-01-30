import os
from hyperspy.io import load

my_path = os.path.dirname(__file__)

example_detector_image = None


def get_detector_image_signal():
    """Get an example signal of a STEM detector

    Example
    -------
    >>> import atomap.api as am
    >>> s = am.example_data.get_detector_image_signal()

    """
    global example_detector_image
    if example_detector_image is None:
        path = os.path.join(my_path, "example_data", "example_detector_image.hspy")
        example_detector_image = load(path)
    s = example_detector_image.deepcopy()
    return s
