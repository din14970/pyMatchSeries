import gc
import os
import shutil

import dask.array as da
import numpy as np
import pytest
from hyperspy.signals import EDSTEMSpectrum, Signal2D

from pymatchseries import matchseries as ms

# numpy dataset
np.random.seed(1001)
imageds = np.random.randint(1, 5, size=(3, 32, 32))
imageds_da = da.from_array(imageds)
hsimage = Signal2D(imageds)
hsimage_da = Signal2D(imageds_da).as_lazy()
hsimage.metadata.General.original_filename = "non/existant/path.emd"
hsimage.metadata.General.title = "dummy"
hsimage_da.metadata.General.original_filename = "non/existant/path.emd"
hsimage_da.metadata.General.title = "dummy"

# spectrum maps
specmap = np.random.randint(1, 5, size=(3, 32, 32, 3))
specmap_da = da.from_array(specmap)
hsspecmap = EDSTEMSpectrum(specmap)
hsspecmap_da = EDSTEMSpectrum(specmap_da).as_lazy()


@pytest.mark.parametrize("data", [imageds, imageds_da, hsimage, hsimage_da])
def test_match_series_create(data):
    mso = ms.MatchSeries(data)
    assert not mso.completed
    assert type(mso.data) == type(data)


@pytest.mark.xfail(raises=ValueError)
def test_match_series_fail():
    ms.MatchSeries()


@pytest.mark.xfail(raises=NotImplementedError)
def test_match_series_typefail():
    ms.MatchSeries(data="won't work")


@pytest.mark.xfail(raises=ValueError)
def test_match_series_dimfail():
    ms.MatchSeries(data=np.ones((4, 4, 4, 4)))


@pytest.mark.xfail(raises=ValueError)
def test_match_series_sizefail():
    ms.MatchSeries(data=np.ones((4, 6, 6)))


@pytest.mark.parametrize(
    "data, f",
    [
        (imageds, 1),
        (imageds_da, 2),
        (hsimage, 3),
        (hsimage_da, 4),
    ],
)
def test_match_series_save_load(data, f):
    mso = ms.MatchSeries(data, path=f"test{f}")
    mso._MatchSeries__prepare_calculation()
    mso.save_data(mso.input_data_file)
    msl = ms.MatchSeries.load(mso.path)
    assert mso.metadata == msl.metadata
    assert mso.configuration == msl.configuration
    assert type(mso.data) == type(msl.data)
    path = mso.path
    del mso
    del msl
    gc.collect()
    shutil.rmtree(path)


@pytest.fixture(scope="module")
def match_series_dummy():
    mso = ms.MatchSeries(imageds)
    mso.run()
    yield mso
    shutil.rmtree(mso.path)


@pytest.mark.parametrize("data", [None, imageds, imageds_da, hsimage, hsimage_da])
def test_match_series_apply_images(data, match_series_dummy):
    defdat = match_series_dummy.get_deformed_images(data)
    if data is None:
        data = match_series_dummy.data
    assert type(data) == type(defdat)


@pytest.mark.parametrize("data", [specmap, specmap_da, hsspecmap, hsspecmap_da])
def test_match_series_apply_spectra(data, match_series_dummy):
    defspec = match_series_dummy.apply_deformations_to_spectra(data)
    defspec2 = match_series_dummy.apply_deformations_to_spectra(data, sum_frames=False)
    assert type(data) == type(defspec)
    assert type(data) == type(defspec2)
