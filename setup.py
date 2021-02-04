from setuptools import setup, find_packages
from itertools import chain

with open("README.md") as f:
    readme = f.read()


# Projects with optional features for building the documentation and running
# tests. From setuptools:
# https://setuptools.readthedocs.io/en/latest/setuptools.html#declaring-extras-optional-features-with-their-own-dependencies
extra_feature_requirements = {
    "doc": ["sphinx >= 3.0.2", "sphinx-rtd-theme >= 0.4.3"],
    "tests": ["pytest >= 5.4", "pytest-cov >= 2.8.1", "coverage >= 5.0"],
}
extra_feature_requirements["dev"] = ["black >= 19.3b0", "pre-commit >= 1.16"] + list(
    chain(*list(extra_feature_requirements.values()))
)

setup(
    name="pyMatchSeries",
    version="0.1.0",
    description=("A python wrapper for the non-rigid-registration "
                 "code match-series"),
    url='https://github.com/din14970/pyMatchSeries',
    author='Niels Cautaerts',
    author_email='nielscautaerts@hotmail.com',
    license='GPL-3.0',
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=['Topic :: Scientific/Engineering :: Physics',
                 'Intended Audience :: Science/Research',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8'],
    keywords='TEM',
    extras_require=extra_feature_requirements,
    packages=find_packages(exclude=["*tests*", "*examples*"]),
    package_data={'': ['pymatchseries/default_parameters.param']},
    include_package_data=True,
    install_requires=[
        'hyperspy>=1.6.1',
        'Pillow',
        'tabulate',
    ],
)
