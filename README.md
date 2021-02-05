<p align="left">
<a href="https://github.com/din14970/pyMatchSeries/actions"><img alt="Actions Status" src="https://github.com/din14970/pyMatchSeries/workflows/build/badge.svg"></a>
<a href="https://pypi.org/project/pyMatchSeries/"><img alt="PyPI" src="https://img.shields.io/pypi/v/pyMatchSeries.svg?style=flat"></a>
<a href='https://coveralls.io/github/din14970/pyMatchSeries?branch=master'><img src='https://coveralls.io/repos/github/din14970/pyMatchSeries/badge.svg?branch=master' alt='Coverage Status' /></a>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://zenodo.org/badge/latestdoi/304592731"><img src="https://zenodo.org/badge/304592731.svg" alt="DOI"></a>
</p>

# Description
This is a tool for non-rigid registration, primarily for atomic resolution STEM images, and is a python wrapper of the [match-series](https://github.com/berkels/match-series) code developed by B. Berkels. When using this tool, please cite the papers mentioned in that repository and the Zenodo reference at the top of this repo. 

The goal of match-series is to remove slow and fast scan noise in STEM image stacks by comparing the various images against each other. The output of the code are X and Y deformation fields for each image in the stack. These deformations can then be applied to stacks of images or to EDX/EELS spectum maps that were acquired frame by frame. The goal of pymatchseries is to facilitate the set-up of the calculation and to work with the results in python. It is intended to use this tool mainly semi-interactively in a Jupyter notebook, see [the example](https://github.com/din14970/pyMatchSeries/blob/master/examples/example.ipynb).

Try it out here:
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/din14970/pyMatchSeries/testing)

To install, simply pip install:
```
$ pip install --user pyMatchSeries
```

Note that, since it directly tries to call the matchSeries binary in a subprocess, **you must compile and/or install match-series on your own**. The program is available via conda install:

```
$ conda install -c conda-forge match-series
```

# Usage
The tool is intended to work well with [hyperspy](https://hyperspy.org/) and relies on it to some extent. The simplified workflow would be:

```python
import hyperspy.api as hs
from pymatchseries import MatchSeries

data = hs.load("path/to/data/file")             # load the image stack
calculation = MatchSeries(data)                 # initialize the calculation
calculation.configuration["lambda"] = 20        # edit configuration
calculation.run()                               # start the calculation
defdata = calculation.get_deformed_images()     # get the "corrected" image stack
```

Supported data inputs are numpy arrays, dask arrays and hyperspy Signal2D objects, both lazy and non-lazy loaded. The deformations can also be
loaded in as a Hyperspy dataset (ComplexSignal2D), and can also be applied to other image datasets and even spectral map
time series. For more details check out the example.

# Changelog

## v0.1.0
* Significantly simplified the API and made code more future proof
* Trying out CI/CD pipelines
