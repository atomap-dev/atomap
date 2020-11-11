import os
from tempfile import TemporaryDirectory
import pytest
import matplotlib.pyplot as plt
import hyperspy.api as hs


@pytest.fixture(autouse=True)
def userguide_doc_setup_teardown(request):
    hs.preferences.General.nb_progressbar = False
    plt.ioff()
    tmp_dir = TemporaryDirectory()
    org_dir = os.getcwd()
    os.chdir(tmp_dir.name)
    yield
    os.chdir(org_dir)
    tmp_dir.cleanup()
