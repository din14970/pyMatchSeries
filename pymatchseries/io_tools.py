"""
Module that includes tools for converting experimental data into the
file structure required for match-series
"""
from PIL import Image
import subprocess
import concurrent.futures as cf
import logging
from pathlib import Path
import os
import numpy as np
import bz2
import hyperspy.api as hs
import sys
from tabulate import tabulate
import h5py
import warnings
from scipy import ndimage

from . import config_tools as ctools

DEFAULT_FILENAME = "./matchseries_calculation.yaml"
DEFAULT_PREFIX = "frame"


def _get_counter(number_of_frames):
    """Calculate minimum # of digits to label frames"""
    return int(np.log10(number_of_frames))+1


def check_emd_contents(filename):
    """Get information about the contents of an emd file"""
    f = hs.load(filename, sum_frames=False, lazy=True,
                load_SI_image_stack=True)
    table = [["Index", "Dataset name", "Data shape"]]
    for j, i in enumerate(f):
        table.append([j, i.metadata.General.title, i.data.shape])
    print(tabulate(table, headers="firstrow"))


def _overwrite_file(fname):
    """ If file exists 'fname', ask for overwriting and return True or False,
    else return True.
    Parameters
    ----------
    fname : str or pathlib.Path
        Directory to check
    Returns
    -------
    bool :
        Whether to overwrite file.
    """
    if Path(fname).is_file():
        message = f"Overwrite '{fname}' (y/n)?\n"
        try:
            answer = input(message)
            answer = answer.lower()
            while (answer != "y") and (answer != "n"):
                print("Please answer y or n.")
                answer = input(message)
            if answer.lower() == "y":
                return True
            elif answer.lower() == "n":
                return False
        except Exception:
            # We are running in the IPython notebook that does not
            # support raw_input
            logging.info(
                "Your terminal does not support raw input. "
                "Not overwriting. "
                "To overwrite the file use `overwrite=True`"
            )
            return False
    else:
        return True


def _overwrite_dir(fname):
    """ If dir exists 'fname', ask for overwriting and return True or False,
    else return True.
    Parameters
    ----------
    fname : str or pathlib.Path
        Directory to check
    Returns
    -------
    bool :
        Whether to overwrite file.
    """
    if Path(fname).is_dir():
        message = f"Overwrite '{fname}' (y/n)?\n"
        try:
            answer = input(message)
            answer = answer.lower()
            while (answer != "y") and (answer != "n"):
                print("Please answer y or n.")
                answer = input(message)
            if answer.lower() == "y":
                return True
            elif answer.lower() == "n":
                return False
        except Exception:
            # We are running in the IPython notebook that does not
            # support raw_input
            logging.info(
                "Your terminal does not support raw input. "
                "Not overwriting. "
                "To overwrite the file use `overwrite=True`"
            )
            return False
    else:
        return True


class MatchSeries(h5py.File):
    """Class representing a persistent Match series calculation"""

    default_filename = "matchseries_calculation"

    @staticmethod
    def new(data, name=None, over_write=None, **kwargs):
        if isinstance(data, str) or isinstance(data, Path):
            file_path = os.path.abspath(data)
            if not os.path.isfile(file_path):
                raise ValueError(f"Data file {file_path} does not exist")
            absfolder_input, basename = os.path.split(file_path)
            prefix_name, ext = os.path.splitext(basename)
            accepted_extensions = [".emd", ".hspy"]
            if ext not in accepted_extensions:
                raise ValueError(f"Only {accepted_extensions} are accepted")
            if name is None:
                name = prefix_name.replace(" ", "_")
        elif isinstance(data, np.ndarray):
            if data.ndim != 3 or data.shape[0] < 2:
                raise ValueError("The data should be in the form of a 3D"
                                 " data cube (data.ndim = 3) and the first "
                                 "axis should contain more than than 1 element"
                                 )
            if name is None:
                warnings.warn("No name was provided to the calculation, "
                              "choosing default: "
                              f"{MatchSeries.default_filename}.")
                name = "matchseries_calculation"
        elif isinstance(data, hs.signals.Signal2D):
            if data.data.ndim != 3 or data.data.shape[0] < 2:
                raise ValueError("The data should be in the form of a 3D"
                                 " data cube (data.ndim = 3) and the first "
                                 "axis should contain more than than 1 element"
                                 )
            if name is None:
                name = data.metadata.General.title
                if not name:
                    warnings.warn("No name was provided and no dataset title "
                                  "was found. Choosing default: "
                                  f"{MatchSeries.default_filename}")
        else:
            raise TypeError(f"The input data type {type(data)} is not"
                            "supported. Supported are file paths, "
                            "numpy arrays and hyperspy Signal2D objects.")
        if not name.endswith(".hdf5"):
            name = name + ".hdf5"
        if os.path.isfile(name):
            warnings.warn(f"The calculation with name {name} already exists!")
            if over_write is None:
                over_write = _overwrite_file(name)
            if over_write:
                os.remove(name)
            else:
                sys.exit(1)
        f = MatchSeries(name)
        f.create_group("data")
        f["data"].attrs["name"] = os.path.splitext(name)[0]
        f["data"].attrs["data_file_path"] = "_None_"
        f["data"].attrs["image_dataset_indexes"] = "_None_"
        f["data"].attrs["config_file_paths"] = "_None_"
        f["data"].attrs["input_image_folder_paths"] = "_None_"
        f["data"].attrs["output_folder_paths"] = "_None_"
        f["data"].attrs["calculation_success"] = "_None_"
        f["data"].attrs["x_size"] = "_None_"
        f["data"].attrs["y_size"] = "_None_"
        f["data"].attrs["x_unit"] = "pixels"
        f["data"].attrs["y_unit"] = "pixels"
        f["data"].attrs["x_scale"] = 1
        f["data"].attrs["y_scale"] = 1
        f._prepare_datasets(data, **kwargs)
        print(f"A new calculation was created with name {name}")
        return f

    @staticmethod
    def load(name=None):
        if name is None:
            warnings.warn("Data and name empty, will try to read default"
                          f" filename: {MatchSeries.default_filename}")
            name = MatchSeries.default_filename + ".hdf5"
        if not name.endswith(".hdf5"):
            name = name + ".hdf5"
        # no data is provided, the file must already exist
        if not os.path.isfile(name):
            raise ValueError(f"{name} not found, can not load datasets.")
        return MatchSeries(name)

    def __init__(self, name, **kwargs):
        super().__init__(name, mode="a", *kwargs)

    def _prepare_datasets(self, data, **kwargs):
        if isinstance(data, str) or isinstance(data, Path):
            if data.endswith(".emd"):
                self._extract_emd(data, **kwargs)
            elif data.endswith(".hspy"):
                self._extract_hspy(data, **kwargs)
            else:
                raise NotImplementedError("This file type not supported.")
        elif isinstance(data, np.ndarray):
            self._extract_npy(data, **kwargs)
        elif isinstance(data, hs.signals.Signal2D):
            self._extract_bare_signal2d(data, **kwargs)
        else:
            raise NotImplementedError("This type of data is not supported.")

    @property
    def _image_dataset_indexes(self):
        return self["data"].attrs["image_dataset_indexes"].tolist()

    @property
    def name(self):
        return self["data"].attrs["name"]

    @property
    def data_file_path(self):
        return self["data"].attrs["data_file_path"]

    @property
    def image_folder_paths(self):
        inx = self["data"].attrs["image_dataset_indexes"]
        vals = self["data"].attrs["input_image_folder_paths"]
        return dict(zip(inx, vals))

    @property
    def calculation_output_paths(self):
        inx = self["data"].attrs["image_dataset_indexes"]
        vals = self["data"].attrs["output_folder_paths"]
        return dict(zip(inx, vals))

    @property
    def config_file_paths(self):
        inx = self["data"].attrs["image_dataset_indexes"]
        vals = self["data"].attrs["config_file_paths"]
        return dict(zip(inx, vals))

    def _extract_signal2d(self, ima, output_folder=None,
                          image_filter=None, **kwargs):
        """
        Extract the images from a hyperspy image stack
        """
        if output_folder is None:
            output_folder = os.path.abspath(f"./{self.name}")
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        # apply filters to the image
        digits = _get_counter(ima.data.shape[0])
        original_dtype = ima.data.dtype
        if image_filter is not None:
            ima.map(image_filter, parallel=True, ragged=False)
        ima.data = ima.data.astype(original_dtype)
        title = "input_images"
        opath = os.path.join(output_folder, title)
        if not os.path.isdir(opath):
            os.makedirs(opath)
        logging.info("Exporting the frames")
        _export_frames(ima, output_folder=opath,
                       prefix=DEFAULT_PREFIX,
                       digits=digits)
        logging.info("finished exporting the frames")
        self["data"].attrs["image_dataset_indexes"] = [0]
        self["data"].attrs["spectroscopy_dataset_indexes"] = []
        self["data"].attrs["input_image_folder_paths"] = [opath]
        cfilename = os.path.join(output_folder,
                                 f"{title}.par")
        pathpattern = os.path.join(
                output_folder,
                f"{title}/{DEFAULT_PREFIX}_%0{digits}d.tiff")
        # already create the folder for the output
        outputpath = os.path.join(output_folder,
                                  f"{title}_results/")
        # level = log2 of image size
        outlevel = int(np.log2(ima.data.shape[-1]))
        if not os.path.isdir(outputpath):
            os.makedirs(outputpath)
        ctools.create_config_file(cfilename, pathpattern=pathpattern,
                                  savedir=outputpath,
                                  preclevel=outlevel,
                                  num_frames=ima.data.shape[0],
                                  **kwargs)
        self["data"].attrs["output_folder_paths"] = [outputpath]
        self["data"].attrs["config_file_paths"] = [cfilename]
        self["data"].attrs["x_size"] = ima.axes_manager[-1].size
        self["data"].attrs["y_size"] = ima.axes_manager[-2].size
        self["data"].attrs["x_unit"] = ima.axes_manager[-1].units
        self["data"].attrs["y_unit"] = ima.axes_manager[-2].units
        self["data"].attrs["x_scale"] = ima.axes_manager[-1].scale
        self["data"].attrs["y_scale"] = ima.axes_manager[-2].scale

    def _extract_hspy(self, file_path, output_folder=None,
                      image_filter=None, **kwargs):
        """
        Extract image frames from a 3D hyperspy dataset
        """
        ima = hs.load(file_path, lazy=True)
        self["data"].attrs["data_file_path"] = file_path
        self._extract_signal2d(ima, output_folder=output_folder,
                               image_filter=image_filter, **kwargs)

    def _extract_npy(self, data, output_folder=None,
                     image_filter=None, **kwargs):
        """
        Extract image frames from a 3D numpy array
        """
        ima = hs.signals.Signal2D(data)
        self["data"].attrs["data_file_path"] = "_None_"
        self._extract_signal2d(ima, output_folder=output_folder,
                               image_filter=image_filter, **kwargs)
        logging.info("Also storing a copy of the data inside the hdf5 file")
        self["data"].create_dataset("data", data=ima.data)

    def _extract_bare_signal2d(self, ima, output_folder=None,
                               image_filter=None, **kwargs):
        """
        Extract image fromes from a 3D hyperspy dataset loaded into memory
        """
        if output_folder is None:
            output_folder = os.path.abspath(f"./{self.name}")
        if not os.path.isdir(output_folder):
            os.makedirs(output_folder)
        opath = os.path.join(output_folder, "input_images.hspy")
        ima.save(opath)
        self["data"].attrs["data_file_path"] = opath
        self._extract_signal2d(ima, output_folder=output_folder,
                               image_filter=image_filter, **kwargs)

    def _extract_emd(self, file_path, output_folder=None,
                     image_dataset_index=None,
                     image_filter=None, **kwargs):
        """
        Extract images and spectrum data from Velox emd files using Hyperspy.

        Creates the correct folder structure that is required by match-series

        Parameters
        ----------
        file_path : str
            path to the emd file
        output_folder : str, optional
            path to the folder where the output files should be saved. Defaults
            to the folder in which the emd file is saved.
        image_dataset_index : int or list of ints, optional
            integer or list of integers to indicate which image datasets must
            be extracted. By default, all are extracted.
        image_filter: callable, optional
            callable that acts on the images to process them before they are
            exported.

        Additional parameters
        ---------------------
        Any value for the MatchSeries config file can be passed here.
        """
        input_path = os.path.abspath(file_path)
        self["data"].attrs["data_file_path"] = input_path
        absfolder_input, basename = os.path.split(input_path)
        prefix_name, ext = os.path.splitext(basename)
        if output_folder is None:
            output_folder = os.path.dirname(input_path)
        elif not os.path.isdir(output_folder):
            logging.warning(f"{output_folder} is not a valid directory")
            logging.warning("Will attempt to create this directory")
            os.makedirs(output_folder)
        output_folder = os.path.abspath(output_folder)
        # read the file
        try:
            f = hs.load(input_path, sum_frames=False, lazy=True,
                        load_SI_image_stack=True)
            imds = [j for j, i in enumerate(f) if
                    isinstance(i, hs.signals.Signal2D) and
                    i.data.ndim == 3]
            spds = [j for j, i in enumerate(f) if
                    isinstance(i, hs.signals.EDSTEMSpectrum) and
                    i.data.ndim == 4]
            logging.info(f"Opened file {input_path}")
            logging.info(f"Found image datasets: {imds}")
            logging.info(f"Found spectroscopy datasets: {spds}")
        except Exception as e:
            raise Exception(f"Something went wrong reading the file: {e}")
        # Images
        # if no dataset is given we extract all of them
        if image_dataset_index is None:
            dsets = imds
        elif isinstance(image_dataset_index, int):
            if image_dataset_index in imds:
                dsets = [image_dataset_index]
            else:
                raise ValueError("The image_dataset_index does not match "
                                 "any image dataset")
        elif isinstance(image_dataset_index, list):
            if set(image_dataset_index) <= set(imds):
                dsets = image_dataset_index
            else:
                raise ValueError("The image_dataset_index list is not a "
                                 "subset of the image dataset indexes.")
        else:
            raise TypeError("image_dataset_index received unexpected type:"
                            f"{type(image_dataset_index)}")
        # the name to prepend to all folders
        pn = prefix_name.replace(" ", "_")
        # store the dataset indexes to the file
        self["data"].attrs["image_dataset_indexes"] = dsets
        # first export all the relevant image datasets
        image_folder_paths = []
        config_file_paths = []
        calculation_results = []
        for k in dsets:
            try:
                # image is the k'th image dataset
                ima = f[k]
                self["data"].attrs["x_size"] = ima.axes_manager[-1].size
                self["data"].attrs["y_size"] = ima.axes_manager[-2].size
                self["data"].attrs["x_unit"] = ima.axes_manager[-1].units
                self["data"].attrs["y_unit"] = ima.axes_manager[-2].units
                self["data"].attrs["x_scale"] = ima.axes_manager[-1].scale
                self["data"].attrs["y_scale"] = ima.axes_manager[-2].scale
                logging.debug(f"Trying to export dataset {k}")
                title = ima.metadata.General.title.replace(" ", "_")
                opath = str(Path(f"{output_folder}/{pn}/{k}_{title}/"))
                if not os.path.isdir(opath):
                    os.makedirs(opath)
                # number of digits required to represent the images
                digits = _get_counter(ima.data.shape[0])
                # apply filters to the image
                original_dtype = ima.data.dtype
                if image_filter is not None:
                    ima.map(image_filter, parallel=True, ragged=False)
                ima.data = ima.data.astype(original_dtype)
                _export_frames(ima, output_folder=opath,
                               prefix=DEFAULT_PREFIX,
                               digits=digits)
                logging.info(f"Saved out the frames to {opath}")
                # also create a config file for all datasets
                # construct the config file
                cfilename = os.path.join(output_folder,
                                         f"{pn}/{k}_{title}.par")
                pathpattern = os.path.join(
                        output_folder,
                        f"{pn}/{k}_{title}/{DEFAULT_PREFIX}_%0{digits}d.tiff")
                # already create the folder for the output
                outputpath = os.path.join(output_folder,
                                          f"{pn}/{k}_{title}_results/")
                # level = log2 of image size
                outlevel = int(np.log2(ima.data.shape[-1]))
                if not os.path.isdir(outputpath):
                    os.makedirs(outputpath)
                ctools.create_config_file(cfilename, pathpattern=pathpattern,
                                          savedir=outputpath,
                                          preclevel=outlevel,
                                          num_frames=ima.data.shape[0],
                                          **kwargs)
                logging.info(
                      f"Dataset {k} was exported to {opath}. A config file "
                      f"{cfilename} was created.")
                image_folder_paths.append(opath)
                config_file_paths.append(cfilename)
                calculation_results.append(outputpath)
            except Exception as e:
                raise e
                logging.warning(f"Dataset {k} was not exported: {e}")
        self["data"].attrs["input_image_folder_paths"] = image_folder_paths
        self["data"].attrs["config_file_paths"] = config_file_paths
        self["data"].attrs["output_folder_paths"] = calculation_results

    @property
    def success(self):
        calcsuc = self["data"].attrs["calculation_success"]
        if not isinstance(calcsuc, str):
            return calcsuc.tolist()
        else:
            return []

    @property
    def summary(self):
        print(f"MatchSeries calculation: {self.name}")
        table = [["Index", "Name", "Calculated?", "Full path"]]
        for i in self._image_dataset_indexes:
            fp = self.image_folder_paths[i]
            name = os.path.split(fp)[-1]
            row = [i, name, i in self.success, fp]
            table.append(row)
        print(tabulate(table, headers="firstrow"))

    def _simulate_calculation(self, index=0):
        if index is None:
            index = self._default_index
        try:
            self.config_file_paths[index]
        except KeyError:
            raise ValueError(f"No configuration was found for dataset {index}")
        print("Starting matchSeries, this can take a while.")
        print("Follow the progress in the terminal window.")
        print("The calculation is done, saving data.")
        success = self.success
        success.append(index)
        self["data"].attrs["calculation_success"] = list(set(success))
        print("Done.")

    @property
    def _default_index(self):
        return self._image_dataset_indexes[0]

    def load_configuration(self, index=None):
        if index is None:
            index = self._default_index
        cfp = self.config_file_paths[index]
        return ctools.load_config(cfp)

    def modify_configuration(self, dic, index=None):
        cfp = self.config_file_paths[index]
        cf = self.load_configuration(index=index)
        for k, v in dic.items():
            cf[k] = v
        cf.save(cfp)

    def calculate_deformations(self, index=None):
        """Run match series for a particular config file"""
        if index is None:
            index = self._default_index
        try:
            config_file = self.config_file_paths[index]
        except KeyError:
            raise ValueError(f"No configuration was found for dataset {index}")
        cmd = ["matchSeries", f"{config_file}"]
        print("Starting matchSeries, this can take a while.")
        print("Follow the progress in the terminal window.")
        try:
            process1 = subprocess.Popen(cmd)
            process1.wait()
        except Exception as e:
            raise OSError("matchSeries was not found on your system. "
                          "Please conda install matchSeries or compile "
                          f"from source. Additional error info: {e}")
        print("The calculation is done, saving data.")
        success = self.success
        success.append(index)
        self["data"].attrs["calculation_success"] = list(set(success))
        print("Done.")

    def _get_stage_bznum(self, index):
        """For extracting the right defx and defy"""
        cf_path = self.config_file_paths[index]
        cf_dic = ctools.load_config(cf_path)
        bznumber = str(cf_dic["stopLevel"]).zfill(2)
        stage = int(cf_dic["numExtraStages"])+1
        return stage, bznumber

    def _get_frame_list(self, index):
        cfn = self.config_file_paths[index]
        cf = ctools.load_config(cfn)
        sf = cf["templateSkipNums"]
        numframes = cf["numTemplates"]
        numoffset = cf["templateNumOffset"]
        numstep = cf["templateNumStep"]
        frames = range(numoffset, numframes, numstep)
        frames = [i for i in frames if i not in sf]
        return frames

    def get_deformations_frame(self, frame_index, image_set_index=None):
        """Return the X and Y deformations as numpy arrays"""
        if image_set_index is None:
            image_set_index = self._default_index
        if image_set_index not in self.success:
            raise ValueError("No successful calculation found for index "
                             f"{image_set_index}")
        try:
            result_folder = self.calculation_output_paths[image_set_index]
        except Exception:
            raise ValueError("This index does not correspond to valid data")
        frms = self._get_frame_list(image_set_index)
        i = frame_index
        result_folder = self.calculation_output_paths[image_set_index]
        stage, bznumber = self._get_stage_bznum(image_set_index)
        if frame_index == 0:
            defX = _loadFromQ2bz(
                    str(Path(f"{result_folder}/stage{stage}/{i}/"
                             f"deformation_{bznumber}_0.dat.bz2")))
            defY = _loadFromQ2bz(
                    str(Path(f"{result_folder}/stage{stage}/{i}/"
                             f"deformation_{bznumber}_1.dat.bz2")))
        elif frame_index <= max(frms):
            defX = _loadFromQ2bz(
                    str(Path(f"{result_folder}/stage{stage}/{i}-r/"
                             f"deformation_{bznumber}_0.dat.bz2")))
            defY = _loadFromQ2bz(
                    str(Path(f"{result_folder}/stage{stage}/{i}-r/"
                             f"deformation_{bznumber}_1.dat.bz2")))
        else:
            raise ValueError("The index is out of bounds")
        return defX, defY

    def load_deformations_as_signal2D(self, index=None):
        """Loads the deformation stack as imaginary X + iY dataset"""
        if index is None:
            index = self._default_index
        frames = self._get_frame_list(index)
        numframes = len(frames)
        newshape = (numframes,
                    self["data"].attrs["x_size"],
                    self["data"].attrs["x_size"],)
        axlist = [
                {
                    "name": "frames",
                    "size": numframes,
                    "navigate": True,
                },
                {
                    "name": "y",
                    "size": self["data"].attrs["y_size"],
                    "units": self["data"].attrs["y_unit"],
                    "scale": self["data"].attrs["y_scale"],
                    "navigate": False,
                },
                {
                    "name": "x",
                    "size": self["data"].attrs["x_size"],
                    "units": self["data"].attrs["x_unit"],
                    "scale": self["data"].attrs["x_scale"],
                    "navigate": False,
                },
                ]
        newds = hs.signals.ComplexSignal2D(np.zeros(newshape), axes=axlist)
        for j, i in enumerate(frames):
            defs = self.get_deformations_frame(i, image_set_index=index)
            def_imag = defs[0] + 1j*defs[1]
            newds.inav[j] = def_imag
        return newds

    def apply_deformations_to_images(self, stack, index=None):
        if index is None:
            index = self._default_index
        frames = self._get_frame_list(index)
        numframes = len(frames)
        newshape = (numframes, *stack.data.shape[-2:])
        axes = stack.axes_manager.as_dictionary()
        axes["axis-0"]["size"] = numframes
        axlist = [axes["axis-0"], axes["axis-1"], axes["axis-2"]]
        newds = hs.signals.Signal2D(np.zeros(newshape), axes=axlist)
        for j, i in enumerate(frames):
            print(f"Processed frame {j+1}/{numframes}")
            defs = self.get_deformations_frame(i, image_set_index=index)
            newds.inav[j] = deform_data(stack.inav[i], defs)
        return newds

    def apply_deformations_to_spectra(self, specmap, index=None):
        if specmap.data.ndim != 4:
            raise ValueError("Spectrum map should be provided as individual "
                             "frames. Please read in with flag "
                             "sum_frames=False.")
        if index is None:
            index = self._default_index
        frames = self._get_frame_list(index)
        numframes = len(frames)
        newshape = (specmap.axes_manager[0].size,
                    specmap.axes_manager[1].size,
                    specmap.axes_manager[3].size)
        axes = specmap.axes_manager.as_dictionary()
        axes["axis-0"]["size"] = numframes
        axlist = [axes["axis-1"], axes["axis-2"], axes["axis-3"]]
        newds = hs.signals.EDSTEMSpectrum(np.zeros(newshape), axes=axlist)
        for j, i in enumerate(frames):
            print(f"Processed frame {j+1}/{numframes}")
            defs = self.get_deformations_frame(i, image_set_index=index)
            to_add = deform_data(specmap.inav[:, :, i].T, defs).T
            newds.data = newds.data + to_add.data
        return newds


def _save_frame_to_file(i, data, path, name, counter, data_format="tiff"):
    """Helper function for multithreading, saving frame i of stack"""
    c = str(i).zfill(counter)
    fp = str(Path(f"{path}/{name}_{c}.{data_format}"))
    frm = data[i]
    try:
        frm = frm.compute()
    except Exception:
        pass
    img = Image.fromarray(frm)
    img.save(fp)
    logging.debug("Wrote out image {name}_{c}")


class _FrameByFrame(object):
    """A pickle-able wrapper for doing a function on all frames of a stack"""
    def __init__(self, do_in_loop, stack, *args, **kwargs):
        self.func = do_in_loop
        self.stack = stack
        self.args = args
        self.kwargs = kwargs

    def __call__(self, index):
        self.func(index, self.stack, *self.args, **self.kwargs)


def _export_frames(stack, output_folder=None, prefix="frame",
                   digits=None, frames=None, multithreading=True,
                   data_format="tiff"):
    """
    Export a 3D data array as individual images
    """
    if frames is None:
        toloop = range(stack.data.shape[0])
    elif isinstance(frames, list):
        toloop = frames
    else:
        raise TypeError("Argument frames must be a list")
    if multithreading:
        with cf.ThreadPoolExecutor() as pool:
            logging.debug("Starting in parallel mode to export")
            pool.map(_FrameByFrame(_save_frame_to_file, stack.data,
                                   output_folder, prefix, digits, data_format),
                     toloop)
    else:
        for i in toloop:
            logging.debug("Exporting frames sequentially")
            _save_frame_to_file(i, stack.data, output_folder, prefix, digits,
                                data_format)


def _loadFromQ2bz(path):
    """
    Opens a bz2 or q2bz file and returns an "image" = the deformations
    """
    filename, file_extension = os.path.splitext(path)
    # bz2 compresses only a single file
    if(file_extension == '.q2bz' or file_extension == '.bz2'):
        # read binary mode, r+b would be to also write
        fid = bz2.open(path, 'rb')
    else:
        fid = open(path, 'rb')  # read binary mode, r+b would be to also write
    # rstrip removes trailing zeros
    binaryline = fid.readline()  # will look like "b"P9\n""
    line = binaryline.rstrip().decode('ascii')  # this will look like "P9"
    if(line[0] != 'P'):
        raise ValueError("Invalid array header, doesn't start with 'P'")
    if(line[1] == '9'):
        dtype = np.float64
    elif(line[1] == '8'):
        dtype = np.float32
    else:
        dtype = None

    if not dtype:
        raise NotImplementedError(
            f"Invalid data type ({line[1]}), only float and "
            "double are supported currently")
    # Skip header = b'# This is a QuOcMesh file of type 9 (=RAW DOUBLE)
    # written 17:36 on Friday, 07 February 2020'
    _ = fid.readline().rstrip()
    # Read width and height
    arr = fid.readline().rstrip().split()
    width = int(arr[0])
    height = int(arr[1])
    # Read max, but be careful not to read more than one new line after max.
    # The binary data could start with a value that is equivalent to a
    # new line.
    max = ""
    while True:
        c = fid.read(1)
        if c == b'\n':
            break
        max = max + str(int(c))

    max = int(max)
    # Read image to vector
    x = np.frombuffer(fid.read(), dtype)
    img = x.reshape(height, width)
    return img


def deform_data(stack, deformations):
    """
    Apply X and Y deformation fields to a stack of images or a spectrum
    map in place

    Parameters
    ----------
    stack : hs.signals.Signal2D
        Stack of images
    deformations : (X, Y), with X and Y same size as images
        Deformation fields, must be same size as  the data

    Returns
    -------
    result : hs.signals.Signal2d
    """
    defX, defY = deformations
    w, h = stack.data.shape[-2:]
    coords = \
        np.mgrid[0:h, 0:w] + np.multiply([defY, defX], (np.max([h, w])-1))

    def mapping(x):
        return ndimage.map_coordinates(x, coords, order=0,
                                       mode="constant")
    return stack.map(mapping, inplace=False, parallel=True, ragged=False)
