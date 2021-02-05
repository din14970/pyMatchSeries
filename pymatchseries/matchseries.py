"""
Module that includes tools for converting experimental data into the
file structure required for match-series
"""
from subprocess import Popen, PIPE, STDOUT
import logging
from pathlib import Path
import os
import numpy as np
import hyperspy.api as hs
from tabulate import tabulate
import warnings
from scipy import ndimage
import uuid
import dask.array as da
from dask import delayed
from dask.diagnostics import ProgressBar
import shutil
import json
import h5py

from pymatchseries import config_tools as ctools
from pymatchseries import io_utils as ioutls

DEFAULT_META_FILE = "metadata.json"
DEFAULT_CONFIG_FILE = "config.par"
DEFAULT_INPUT_FOLDER = "input/"
DEFAULT_OUTPUT_FOLDER = "output/"
DEFAULT_PREFIX = "frame"
DEFAULT_DATA_FILE = "data"


def _get_counter(number_of_frames):
    """Calculate minimum # of digits to label frames"""
    return int(np.log10(number_of_frames)) + 1


def _is_valid_calculation(path):
    metafile = os.path.join(path, DEFAULT_META_FILE)
    configfile = os.path.join(path, DEFAULT_CONFIG_FILE)
    inputfolder = os.path.join(path, DEFAULT_INPUT_FOLDER)
    outputfolder = os.path.join(path, DEFAULT_OUTPUT_FOLDER)
    hdf5 = os.path.join(path, DEFAULT_DATA_FILE + ".hdf5")
    hspy = os.path.join(path, DEFAULT_DATA_FILE + ".hspy")
    hasmetadata = os.path.isfile(metafile)
    hasconfig = os.path.isfile(configfile)
    hasinputfolder = os.path.isdir(inputfolder)
    hasoutputfolder = os.path.isdir(outputfolder)
    hasdatafile = os.path.isfile(hdf5) or os.path.isfile(hspy)
    condition = (
        hasmetadata and hasconfig and hasinputfolder and hasoutputfolder and hasdatafile
    )
    return condition


def _load_metadata(path):
    metafile = os.path.join(path, DEFAULT_META_FILE)
    with open(metafile) as f:
        return json.load(f)


def _load_configuration(path):
    configfile = os.path.join(path, DEFAULT_CONFIG_FILE)
    return ctools.load_config(configfile)


def _calculation_completed(path):
    meta = _load_metadata(path)
    return meta["completed"]


def _get_raw_data(data):
    if isinstance(data, hs.signals.BaseSignal):
        return data.data
    else:
        return data


class MatchSeries:
    """Class representing a Match series calculation"""

    def __init__(self, data=None, path=None, **kwargs):
        if data is not None:
            # we provide data, we try to create new
            self.__setup_new_calculation(data, path, **kwargs)
        elif path is not None:
            # data is none, we try to load
            self.__load_calculation(path)
        else:
            raise ValueError(
                "Either data and/or a path to a calculation must be provided"
            )

    def __load_calculation(self, path):
        if not _is_valid_calculation(path):
            raise ValueError(f"Matchseries data {path} does not seem valid")
        self.path = path
        self.__metadata = _load_metadata(path)
        self.configuration = _load_configuration(path)
        self.__load_data()

    @property
    def completed(self):
        return self.metadata["completed"]

    def __setup_new_calculation(self, data, path, **kwargs):
        self.__data = data
        self.__metadata = {}
        if isinstance(data, np.ndarray) or isinstance(data, da.core.Array):
            self.__metadata["x_name"] = "x"
            self.__metadata["x_scale"] = 1
            self.__metadata["x_offset"] = 0
            self.__metadata["x_unit"] = "pixels"
            self.__metadata["y_name"] = "y"
            self.__metadata["y_scale"] = 1
            self.__metadata["y_offset"] = 0
            self.__metadata["y_unit"] = "pixels"
            self.__metadata["input_type"] = "array"
            EXT = "hdf5"
            if isinstance(data, np.ndarray):
                self.__metadata["lazy"] = False
            else:
                self.__metadata["lazy"] = True
        elif isinstance(data, hs.signals.Signal2D):
            self.__metadata["x_name"] = (
                str(data.axes_manager[-2].name).replace("<", "").replace(">", "")
            )
            self.__metadata["x_scale"] = data.axes_manager[-2].scale
            self.__metadata["x_offset"] = data.axes_manager[-2].offset
            self.__metadata["x_unit"] = (
                str(data.axes_manager[-2].units).replace("<", "").replace(">", "")
            )
            self.__metadata["y_name"] = (
                str(data.axes_manager[-1].name).replace("<", "").replace(">", "")
            )
            self.__metadata["y_scale"] = data.axes_manager[-1].scale
            self.__metadata["y_offset"] = data.axes_manager[-1].offset
            self.__metadata["y_unit"] = (
                str(data.axes_manager[-1].units).replace("<", "").replace(">", "")
            )
            self.__metadata["input_type"] = "hyperspy"
            EXT = "hspy"
            self.__metadata["lazy"] = data._lazy
        else:
            raise NotImplementedError(
                f"The input data type {type(data)} is not"
                "supported. Supported are "
                "numpy arrays, dask arrays, and hyperspy Signal2D objects."
            )
        if path is None:
            try:
                # this will only work for some hspy datasets with the right metadata
                p, fn = os.path.split(data.metadata.General.original_filename)
                filename, _ = os.path.splitext(fn)
                title = data.metadata.General.title
                path = f"./{filename}_{title}/"
            except AttributeError:
                path = "./" + str(uuid.uuid4()).replace("-", "") + "/"

        if self.image_data.ndim != 3 or self.image_data.shape[0] < 2:
            raise ValueError(
                "The data should be in the form of a 3D"
                " data cube (data.ndim = 3) and the first "
                "axis should contain more than than 1 element"
            )
        self.path = os.path.abspath(path)
        # relative to paths
        self.__metadata["metadata_file_path"] = DEFAULT_META_FILE
        self.__metadata["input_folder_path"] = DEFAULT_INPUT_FOLDER
        self.__metadata["output_folder_path"] = DEFAULT_OUTPUT_FOLDER
        self.__metadata["config_file_path"] = DEFAULT_CONFIG_FILE
        self.__metadata["input_data_file"] = f"{DEFAULT_DATA_FILE}.{EXT}"
        self.__metadata["prefix"] = DEFAULT_PREFIX
        self.__metadata["completed"] = False
        self.__metadata["x_dim"] = self.image_data.shape[1]
        self.__metadata["y_dim"] = self.image_data.shape[2]
        self.__metadata["z_dim"] = self.image_data.shape[0]
        self.__metadata["digits"] = _get_counter(self.image_data.shape[0])
        pathpattern = os.path.join(
            self.metadata["input_folder_path"],
            f"frame_%0{self.metadata['digits']}d.tiff",
        )
        # level = log2 of image size
        outlevel = np.log2(self.image_data.shape[-1])
        if outlevel != int(outlevel):
            raise ValueError("Images must be square with side length a factor of 2")
        outlevel = int(outlevel)
        # create a default configuration
        self.configuration = ctools.get_configuration(
            templateNamePattern=pathpattern,
            saveDirectory=self.metadata["output_folder_path"],
            precisionLevel=outlevel,
            numTemplates=self.image_data.shape[0],
            **kwargs,
        )

    @property
    def data(self):
        return self.__data  # the data should not be modified

    @property
    def image_data(self):
        return _get_raw_data(self.data)

    @property
    def metadata(self):
        return self.__metadata  # the metadata should not be modified

    def run(self):
        self.__prepare_calculation()
        self.save_data(self.input_data_file)
        self.__run_match_series()
        self.__metadata["completed"] = True
        self.__update_metadata_file()

    def save_data(self, path):
        """Writes the data in the right format to disk"""
        if self.metadata["input_type"] == "hyperspy":
            self.data.save(path)
        else:
            if isinstance(self.data, da.core.Array):
                da.to_hdf5(path, f"/{DEFAULT_DATA_FILE}", self.data)
            else:
                with h5py.File(path, "w") as f:
                    f.create_dataset(f"{DEFAULT_DATA_FILE}", data=self.data)

    def __load_data(self):
        lazy = self.metadata["lazy"]
        path = self.input_data_file
        if self.metadata["input_type"] == "hyperspy":
            self.__data = hs.load(path, lazy=lazy)
        else:
            with h5py.File(path, "r") as f:
                if lazy:
                    self.__data = da.from_array(f[f"{DEFAULT_DATA_FILE}"])
                else:
                    self.__data = f[f"{DEFAULT_DATA_FILE}"][:]

    def __update_metadata_file(self):
        with open(self.metadata_file_path, "w") as md:
            json.dump(self.metadata, md)

    @property
    def input_folder_path(self):
        return os.path.join(self.path, self.metadata["input_folder_path"])

    @property
    def output_folder_path(self):
        return os.path.join(self.path, self.metadata["output_folder_path"])

    @property
    def config_file_path(self):
        return os.path.join(self.path, self.metadata["config_file_path"])

    @property
    def input_data_file(self):
        return os.path.join(self.path, self.metadata["input_data_file"])

    @property
    def metadata_file_path(self):
        return os.path.join(self.path, self.metadata["metadata_file_path"])

    def __prepare_calculation(self):
        if os.path.isdir(self.path):
            warnings.warn(f"The calculation {self.path} already exists!")
            if ioutls.overwrite_dir(self.path):
                shutil.rmtree(self.path)
            else:
                return
        # creating the folder structure
        os.makedirs(self.path)
        os.makedirs(self.input_folder_path)
        os.makedirs(self.output_folder_path)
        # extracting the data and writing the relevant files
        logging.info("Exporting the frames")
        ioutls.export_frames(
            self.image_data,
            folder=self.input_folder_path,
            prefix=self.metadata["prefix"],
            digits=self.metadata["digits"],
            multithreading=False,
        )
        logging.info("Finished exporting the frames")
        self.configuration.save(self.config_file_path)
        self.__update_metadata_file()

    def __run_match_series(self):
        """Run match series using the config file and print all output"""
        # from https://github.com/takluyver/rt2-workshop-jupyter/blob/e7fde6565e28adf31a0f9003094db70c3766bd6d/Subprocess%20output.ipynb
        cmd = ["matchSeries", f"{self.config_file_path}"]
        p = Popen(cmd, stdout=PIPE, stderr=STDOUT, cwd=self.path)
        while True:
            output = p.stdout.read1(1024).decode("utf-8")
            print(output, end="")
            if p.poll() is not None:
                break
        if p.returncode != 0:
            raise Exception(("Exited with error code:", p.returncode))

    @staticmethod
    def discover(path=".", recursive=False):
        """Find calculations in a certain folder"""
        if recursive:
            paths = [i[0] for i in os.walk(path)]
        else:
            paths = [path]
        table = []
        for path in paths:
            path = os.path.expanduser(path)
            nodes = [os.path.join(path, i) for i in os.listdir(path=path)]
            dirs = [i for i in nodes if os.path.isdir(i)]
            for folder in dirs:
                if _is_valid_calculation(folder):
                    meta = _load_metadata(folder)
                    conf = _load_configuration(folder)
                    completed = meta["completed"]
                    shape = (meta["x_dim"], meta["y_dim"])
                    images = len(conf._get_frame_list())
                    table.append([folder, completed, images, shape])
        tabular = tabulate(
            table, headers=["Path", "Completed?", "# Images", "(width, height)"]
        )
        print(tabular)

    @staticmethod
    def load(path):
        """Load an existing MatchSeries calculation"""
        ms = MatchSeries(path=path)
        return ms

    def __is_existing_frame(self, frame):
        frms = self.configuration._get_frame_list()
        return frame in frms

    def __load_deformation(self, frame_index, axis="x", refined=True):
        """Instruction to load a single array lazily"""
        result_folder = self.output_folder_path
        stage, bznumber = self.configuration._get_stage_bznum()
        xr = "-r" if (frame_index != 0 and refined) else ""
        ax = 0 if axis == "x" else 1
        path = str(
            Path(
                f"{result_folder}/stage{stage}/{frame_index}{xr}/"
                f"deformation_{bznumber}_{ax}.dat.bz2"
            )
        )
        loader = delayed(ioutls._loadFromQ2bz)(path)
        shape = (self.metadata["y_dim"], self.metadata["x_dim"])
        deformation = da.from_delayed(loader, shape, dtype=float)
        return deformation

    def __load_deformations_data_lazy(self, refined=True):
        """Return all deformations as dask array"""
        dxs = []
        dys = []
        for i in self.configuration._get_frame_index_iterator():
            dx = self.__load_deformation(i, "x", refined)
            dy = self.__load_deformation(i, "y", refined)
            dxs.append(dx)
            dys.append(dy)
        defx = da.stack(dxs, axis=0)
        defy = da.stack(dys, axis=0)
        return defx, defy

    def import_deformations(self, lazy=False):
        """
        Loads the deformation stack as imaginary X + iY hyperspy dataset
        """
        if not self.completed:
            raise Exception("The deformations have not yet been calculated")
        defx, defy = self.__load_deformations_data_lazy()
        def_imag = defx + 1j * defy
        axlist = self.__get_default_axlist(def_imag.shape[0])
        newds = hs.signals.ComplexSignal2D(def_imag, axes=axlist).as_lazy()
        if not lazy:
            newds.compute()
        return newds

    def __get_default_axlist(self, numframes):
        axlist = [
            {
                "name": "frames",
                "size": numframes,
                "navigate": True,
            },
            {
                "name": self.metadata["y_name"],
                "size": self.metadata["y_dim"],
                "units": self.metadata["y_unit"],
                "scale": self.metadata["y_scale"],
                "offset": self.metadata["y_offset"],
                "navigate": False,
            },
            {
                "name": self.metadata["x_name"],
                "size": self.metadata["x_dim"],
                "units": self.metadata["x_unit"],
                "scale": self.metadata["x_scale"],
                "offset": self.metadata["x_offset"],
                "navigate": False,
            },
        ]
        return axlist

    def __is_valid_data(self, data):
        """Check whether data is the same shape as the calculation"""
        frames = self.configuration._get_frame_list()
        maxframes = np.max(frames)
        xdim = self.metadata["x_dim"]
        ydim = self.metadata["y_dim"]
        raw = _get_raw_data(data)
        return (
            (raw.shape[-2] == ydim)
            and (raw.shape[-1] == xdim)
            and (raw.shape[0] >= maxframes)
        )

    def get_deformed_images(self, data=None, **kwargs):
        """
        Return a stack of images with the deformations applied
        """
        if data is None:
            data = self.data
        else:
            if not self.__is_valid_data(data):
                raise ValueError("The data is not the same shape as the deformations")
        raw = _get_raw_data(data)
        chunks = ("auto", -1, -1)
        if isinstance(raw, np.ndarray):
            dt = da.from_array(raw, chunks=chunks)
        elif isinstance(raw, da.core.Array):
            dt = raw.rechunk(chunks)
        else:
            raise TypeError("Unexpected data type")
        frames = self.configuration._get_frame_list()
        dt = dt[frames]
        defx, defy = self.__load_deformations_data_lazy()
        defx = defx.rechunk(dt.chunks)
        defy = defy.rechunk(dt.chunks)
        defdt = dt.map_blocks(_map_deform_image, defx, defy, dtype=float)
        if isinstance(data, da.core.Array):
            return defdt
        elif isinstance(data, np.ndarray):
            with ProgressBar():
                return defdt.compute(**kwargs)
        elif isinstance(data, hs.signals.Signal2D):
            axes = data.axes_manager.as_dictionary()
            axes["axis-0"]["size"] = defdt.shape[0]
            axlist = [axes["axis-0"], axes["axis-1"], axes["axis-2"]]
            if data._lazy:
                newds = hs.signals.Signal2D(
                    defdt,
                    axes=axlist,
                    metadata=data.metadata.as_dictionary(),
                    original_metadata=data.original_metadata.as_dictionary(),
                )
                return newds.as_lazy()
            else:
                with ProgressBar():
                    defdt = defdt.compute(**kwargs)
                newds = hs.signals.Signal2D(
                    defdt,
                    axes=axlist,
                    metadata=data.metadata.as_dictionary(),
                    original_metadata=data.original_metadata.as_dictionary(),
                )
                return newds
        else:
            raise TypeError(
                "Data must be numpy or dask array " "or a hyperspy 2D signal"
            )

    def __is_valid_specmap(self, specmap):
        frames = self.configuration._get_frame_list()
        maxframes = np.max(frames)
        specmap = _get_raw_data(specmap)
        condition = (
            specmap.ndim == 4
            and specmap.shape[2] == self.metadata["x_dim"]
            and specmap.shape[1] == self.metadata["y_dim"]
            and specmap.shape[0] > maxframes
        )
        return condition

    def apply_deformations_to_spectra(self, specmap, sum_frames=True, **kwargs):
        """
        Apply deformations to a spectrum time series acquired simultaneously
        as the images

        Parameters
        ----------
        specmap: numpy array, dask array or hyperspy EDS spectrum
            The spectral data must be of shape (frames, x_dim, y_dim, channels)
        sum_frames: bool
            Whether to return all the deformed frames or the sum of the
            deformed frames
        **kwargs: passed to dask's compute function

        Returns
        -------
        defmap: numpy array, dask array or hyperspy EDS spectrum
            The deformed spectral data of the same type as the input.
        """
        if not self.__is_valid_specmap(specmap):
            raise TypeError(
                "Must supply a correctly sized 4D spectrum. "
                "Make sure to import with flag sum_frames=False."
            )
        raw = _get_raw_data(specmap)
        chunks = ("auto", -1, -1, "auto")
        if isinstance(raw, np.ndarray):
            dt = da.from_array(raw, chunks=chunks)
        elif isinstance(raw, da.core.Array):
            dt = raw.rechunk(chunks)
        else:
            raise TypeError("Unexpected data type")
        frames = self.configuration._get_frame_list()
        dt = dt[frames]
        defx, defy = self.__load_deformations_data_lazy()
        defx = da.stack([defx] * dt.shape[-1], axis=-1)
        defy = da.stack([defy] * dt.shape[-1], axis=-1)
        defx = defx.rechunk(dt.chunks)
        defy = defy.rechunk(dt.chunks)
        logging.info(f"data: {dt.shape}, {dt.chunksize}")
        logging.info(f"dx: {defx.shape}, {defx.chunksize}")
        logging.info(f"dy: {defy.shape}, {defy.chunksize}")
        defdt = dt.map_blocks(_map_deform_spectra, defx, defy, dtype=float)
        if sum_frames:
            defdt = defdt.sum(axis=0)
        if isinstance(specmap, da.core.Array):
            return defdt
        elif isinstance(specmap, np.ndarray):
            with ProgressBar():
                return defdt.compute(**kwargs)
        elif isinstance(specmap, hs.signals.BaseSignal):
            axes = specmap.axes_manager.as_dictionary()
            axes["axis-0"]["size"] = defdt.shape[0]
            axlist = [axes["axis-0"], axes["axis-1"], axes["axis-2"]]
            if specmap._lazy:
                newds = hs.signals.EDSTEMSpectrum(
                    defdt,
                    axes=axlist,
                    metadata=specmap.metadata.as_dictionary(),
                    original_metadata=specmap.original_metadata.as_dictionary(),
                )
                return newds.as_lazy()
            else:
                with ProgressBar():
                    defdt = defdt.compute(**kwargs)
                newds = hs.signals.EDSTEMSpectrum(
                    defdt,
                    axes=axlist,
                    metadata=specmap.metadata.as_dictionary(),
                    original_metadata=specmap.original_metadata.as_dictionary(),
                )
                return newds
        else:
            raise TypeError(
                "Data must be numpy or dask array " "or a hyperspy 2D signal"
            )


def deform_image(image, defX, defY):
    """
    Apply X and Y deformation fields to an image

    Parameters
    ----------
    image : 2D numpy array
        2D array representing image images
    defX : 2D numpy array
        X deformation of the image
    defY : 2D numpy array
        Y deformation field of the image

    Returns
    -------
    result : 2D numpy array
        2D array representing the deformed image
    """
    h, w = image.shape[-2:]
    coords = np.mgrid[0:h, 0:w] + np.multiply([defY, defX], (np.max([h, w]) - 1))
    return ndimage.map_coordinates(image, coords, order=0, mode="constant")


def _map_deform_image(image_chunk, defX_chunk, defY_chunk):
    """
    Apply deform_image over a chunk of images
    """
    defimg = np.empty(image_chunk.shape, dtype=float)
    for i in range(defimg.shape[0]):
        defimg[i] = deform_image(image_chunk[i], defX_chunk[i], defY_chunk[i])
    return defimg


def _map_deform_spectra(spectra_chunk, defX_chunk, defY_chunk):
    """
    Apply map_deform_image over chunk of 4D spectrum
    """
    def_spec = np.empty(spectra_chunk.shape, dtype=float)
    for i, j in np.ndindex(spectra_chunk.shape[0], spectra_chunk.shape[-1]):
        image = spectra_chunk[i, :, :, j]
        def_spec[i, :, :, j] = deform_image(
            image,
            defX_chunk[i, :, :, j],
            defY_chunk[i, :, :, j],
        )
    return def_spec
