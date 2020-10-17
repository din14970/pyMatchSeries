from setuptools import setup, find_packages

with open("README.md") as f:
    readme = f.read()

setup(
    name="pyMatchSeries",
    version="0.0.1",
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
    packages=find_packages(exclude=["*tests*", "*examples*"]),
    package_data={'': ['pymatchseries/default_parameters.param']},
    include_package_data=True,
    install_requires=[
        'hyperspy',
        'Pillow',
        'tabulate',
    ],
)
