from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("requirements.in") as f:
    requirements = f.read.splitlines()

with open("dev-requirements.in") as f:
    dev_requirements = f.read.splitlines()

# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
extra_feature_requirements = {
    "dev": dev_requirements,
    "gpu": ["cupy"],
}

setup(
    name="pyMatchSeries",
    version="0.3.0",
    description=(
        "A python implementation of joint-non-rigid-registration "
        "and wrapper of match-series."
    ),
    url="https://github.com/din14970/pyMatchSeries",
    author="Niels Cautaerts",
    author_email="nielscautaerts@hotmail.com",
    license="GPL-3.0",
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Topic :: Scientific/Engineering :: Physics",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    keywords="TEM",
    extras_require=extra_feature_requirements,
    packages=find_packages(exclude=["*tests*", "*examples*"]),
    package_data={"": ["pymatchseries/default_parameters.param"]},
    entry_points={"hyperspy.extensions": "pyMatchSeries = pyMatchSeries"},
    include_package_data=True,
    install_requires=requirements,
)
