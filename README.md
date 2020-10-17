# Description
This is a tool for non-rigid registration, primarily for atomic resolution STEM images, and is a python wrapper of the [match-series](https://github.com/berkels/match-series) code developed by berkels. The goal of match-series is to remove slow and fast scan noise in STEM image stacks by comparing the various images against each other. The output of the code are X and Y deformation fields for each image in the stack. These deformations can then be applied to stacks of images or to EDX/EELS spectum maps that were acquired frame by frame. The goal of pymatchseries is to facilitate the set-up of the calculation and to work with the results in python. It is intended to use this tool mainly semi-interactively in a Jupyter notebook, see [the example](examples/example.ipynb).

To install, simply pip install:
```
$ pip install --user pymatchseries
```

Note that, since it directly tries to call the matchSeries binary in a subprocess, **you must compile and install match-series on your own**. On Linux, the program is available via conda install:

```
$ conda install -c conda-forge match-series
```

On other platforms, please clone the match-series source code and compile.
