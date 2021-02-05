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


def test_init():
    testdict = {"foo": "bar", "batman": 1}
    c = ctools.config_dict(testdict)
    assert isinstance(c, ctools.config_dict)


def test_get():
    testdict = {"foo": "bar", "batman": 1}
    c = ctools.config_dict(testdict)
    assert c.foo == "bar"


def test_set():
    testdict = {"foo": "bar", "batman": 1}
    c = ctools.config_dict(testdict)
    c.foo == "foo"
    assert c.foo == "bar"


@pytest.mark.xfail(raises=KeyError)
def test_setfail():
    testdict = {"foo": "bar", "batman": 1}
    c = ctools.config_dict(testdict)
    c.superman == "foo"


def test_getitem():
    testdict = {"foo": "bar", "batman": "1", "superman": "1.1", "joker": "{ 1 3 135 }"}
    c = ctools.config_dict(testdict)
    assert c["foo"] == "bar"
    assert c["batman"] == 1
    assert c["superman"] == 1.1
    assert c["joker"] == [1, 3, 135]


def test_setitem():
    testdict = {"foo": "bar", "batman": "1", "superman": "1.1", "joker": "{ 1 3 135 }"}
    c = ctools.config_dict(testdict)
    c["foo"] = "bla"
    c["batman"] = True
    c["joker"] = [1, 3, 135]


@pytest.mark.xfail(raises=KeyError)
def test_setitemfail():
    testdict = {"foo": "bar", "batman": "1", "superman": "1.1", "joker": "{ 1 3 135 }"}
    c = ctools.config_dict(testdict)
    c["noexist"] = 1


def test_bznum():
    cf = ctools.load_config()
    s, bz = cf._get_stage_bznum()
    assert bz == "08"
    assert s == 3


def test_framelist():
    cf = ctools.load_config()
    cf.templateSkipNums = [3, 9]
    cf.numTemplates = 10
    cf.templateNumOffset = 1
    cf.templateNumStep = 2
    frms = cf._get_frame_list()
    assert frms == [1, 5, 7]


def test_frameindexgen():
    cf = ctools.load_config()
    cf.templateSkipNums = [3, 9]
    cf.numTemplates = 10
    cf.templateNumOffset = 1
    cf.templateNumStep = 2
    frms = cf._get_frame_index_iterator()
    assert list(frms) == [0, 1, 2]
