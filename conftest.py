import pytest

def pytest_addoption(parser):
    parser.addoption("--env", action="store", default="test", help="nogui")

def pytest_configure(config):
    config.addinivalue_line("markers", "skip_test_if_env(env): this mark skips the tests for the given env")

def pytest_runtest_setup(item):
    env_names = [mark.args[0] for mark in item.iter_markers(name="skip_test_if_env")]
    if env_names:
        if item.config.getoption("--env") in env_names:
            pytest.skip("Test skipped because env is {!r}".format(env_names))
