from pymatchseries import config_tools as ctools
import os
import pytest


def test_load_config():
    ctools.load_config()


def test_write_config():
    dummy = ctools.load_config()
    ctools.write_config(dummy, "test_config.par")
    assert os.path.isfile("test_config.par")
    os.remove("test_config.par")


def test_get_configuration():
    c = ctools.get_configuration("test", "test", 2, 1, [1, 2, 3], preSmoothSigma=1)
    assert isinstance(c, ctools.config_dict)


def test_create_config_file():
    ctools.create_config_file(
        "test.par", "test", "test", 2, 1, [1, 2, 3], preSmoothSigma=1
    )
    assert os.path.isfile("test.par")
    os.remove("test.par")


class TestConfigDict:
    def test_init(self):
        testdict = {"foo": "bar", "batman": 1}
        c = ctools.config_dict(testdict)
        assert isinstance(c, ctools.config_dict)

    def test_get(self):
        testdict = {"foo": "bar", "batman": 1}
        c = ctools.config_dict(testdict)
        assert c.foo == "bar"

    def test_set(self):
        testdict = {"foo": "bar", "batman": 1}
        c = ctools.config_dict(testdict)
        c.foo == "foo"
        assert c.foo == "bar"
