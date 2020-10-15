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
import re
import bz2
import hyperspy.api as hs
import sys
from tabulate import tabulate
import h5py

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


def overwrite_dir(fname):
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


path_lists = [
        "image_folder_paths",
        "image_hspy_paths",
        "spectrum_hspy_paths",
        "calculation_output_paths",
        "config_file_paths",
        "results_paths",
        "other_data",
        ]


class MatchSeries(h5py.File):
    """Class representing a Match series calculation"""
    def __init__(self, name, mode="a"):
        if not name.endswith(".hdf5"):
            name = name + ".hdf5"
        if not Path(name).is_file():
            newfile = True
        else:
            newfile = False
        super().__init__(name, mode)
        if newfile:
            for i in path_lists:
                self.create_group(i)

    @property
    def image_folder_paths(self):
        return dict(self["image_folder_paths"].attrs)

    @property
    def image_hspy_paths(self):
        return dict(self["image_hspy_paths"].attrs)

    @property
    def spectrum_hspy_paths(self):
        return dict(self["spectrum_hspy_paths"].attrs)

    @property
    def calculation_output_paths(self):
        return dict(self["calculation_output_paths"].attrs)

    @property
    def config_file_paths(self):
        return dict(self["config_file_paths"].attrs)

    @property
    def results_paths(self):
        return dict(self["results_paths"].attrs)

    @property
    def other_data(self):
        return dict(self["other_data"].attrs)

    def extract_hspy(self, file_path, output_folder=None):
        """
        Extract image frames from a 3D hyperspy dataset
        """
        pass

    def extract_npy(self, data, output_folder=None):
        """
        Extract image frames from a 3D numpy array and
        """
        pass

    def extract_emd(self, file_path, output_folder=None,
                    image_dataset_index=None, spectrum_dataset_index=None,
                    image_filter=None, over_write=None, **kwargs):
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
        spectrum_dataset_index : int or list of ints, optional
            integer or list of integers to indicate which spectrumstream
            datasets must be extracted. By default, all are extracted.
        image_filter: callable, optional
            callable that acts on the images to process them before they are
            exported.
        over_write: bool, optional
            if data was already extracted from this file, over_write will write
            over these datasets without a question. If false it will not
            overwrite datasets. If None, it will pose the question if a
            conflict is found.

        Additional parameters
        ---------------------
        Any value for the MatchSeries config file can be passed here.
        """
        input_path = os.path.abspath(file_path)
        absfolder_input, basename = os.path.split(input_path)
        prefix_name, ext = os.path.splitext(basename)
        if not os.path.isfile(input_path) and ext == "emd":
            raise ValueError(f"{input_path} is not a valid emd file path")
        self["other_data"].attrs["original_file_path"] = input_path
        if output_folder is None:
            output_folder = os.path.dirname(input_path)
        elif not os.path.isdir(output_folder):
            logging.error(f"{output_folder} is not a valid directory")
            logging.error("Will attempt to create this directory")
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
            logging.debug(f"Opened file {input_path}")
            logging.debug(f"Found image datasets: {imds}")
            logging.debug(f"Found spectroscopy datasets: {spds}")
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
        # build up list for spectral dataset indexes
        if spectrum_dataset_index is None:
            sdsets = spds
        elif isinstance(spectrum_dataset_index, int):
            if spectrum_dataset_index in spds:
                sdsets = [spectrum_dataset_index]
            else:
                raise ValueError("The spectrum_dataset_index does not match "
                                 "any spectrum dataset")
        elif isinstance(spectrum_dataset_index, list):
            if set(spectrum_dataset_index) <= set(spds):
                sdsets = spectrum_dataset_index
            else:
                raise ValueError("The spectrum_dataset_index list is not a "
                                 "subset of the spectrum dataset indexes.")
        else:
            raise TypeError("spectrum_dataset_index received unexpected type:"
                            f"{type(spectrum_dataset_index)}")
        # the name to prepend to all folders
        pn = prefix_name.replace(" ", "_")
        # first export all the relevant image datasets
        for k in dsets:
            try:
                # image is the k'th image dataset
                ima = f[k]
                logging.debug(f"Trying to export dataset {k}")
                title = ima.metadata.General.title.replace(" ", "_")
                opath = str(Path(f"{output_folder}/{pn}/{k}_{title}/"))
                if not os.path.isdir(opath):
                    os.makedirs(opath)
                else:
                    # if the directory already exists skip
                    if over_write is None:
                        over_write = overwrite_dir(opath)
                    if over_write is False:
                        continue
                # number of digits required to represent the images
                digits = _get_counter(ima.data.shape[0])
                # apply filters to the image
                original_dtype = ima.data.dtype
                if image_filter is not None:
                    ima.map(image_filter, parallel=True, ragged=False)
                ima.data = ima.data.astype(original_dtype)
                # save the raw dataset as a hyperspy object
                hyperspy_filename = f"{output_folder}/{pn}/{k}_{title}.hspy"
                ima.save(hyperspy_filename, overwrite=True)
                logging.debug(f"Saved hyperspy object to {hyperspy_filename}")
                export_frames(ima, output_folder=opath,
                              prefix=DEFAULT_PREFIX,
                              digits=digits)
                logging.debug(f"Saved out the frames to {opath}")
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
                logging.debug(
                      f"Dataset {k} was exported to {opath}. A config file "
                      f"{cfilename} was created.")
                self["image_folder_paths"].attrs[str(k)] = opath
                self["image_hspy_paths"].attrs[str(k)] = hyperspy_filename
                self["calculation_output_paths"].attrs[str(k)] = outputpath
                self["config_file_paths"].attrs[str(k)] = cfilename
            except Exception as e:
                raise e
                logging.warning(f"Dataset {k} was not exported: {e}")
        # Spectrumstreams
        # are there any in the dataset?
        if not sdsets:
            logging.debug(
                    "There is no spectral data to consider in this dataset")
            sys.exit(1)
        for k in sdsets:
            try:
                spec = f[k]
                logging.debug(f"Working on dataset {k}")
                title = spec.metadata.General.title.replace(" ", "_")
                # hyperspy_filename = f"{output_folder}/{pn}/{k}_{title}.hspy"
                # spec.save(hyperspy_filename)
                # logging.debug("Exported stream to hyperspy object")
            except Exception as e:
                logging.warning(f"Dataset {k} was not exported: {e}")
            self["spectrum_hspy_paths"].attrs[str(k)] = hyperspy_filename

    def run(self, index):
        """Run match series for a particular config file"""
        cmd = [str("matchSeries"), f"{config_file}", ">", "output.log"]
        process1 = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        process1.wait()
        return read_config_file(config_file)["saveDirectory"]

    def get_deformations(self, image_set_index, frame_index):
        pass

    def apply_deformations(self, result_index=0, image_index=None,
                           spectra_index=None, overwrite=None):
        """
        Apply the deformations calculated by match-series to images and
        optionally spectra. The resulting deformed images and spectra are
        written out to a folder for later import if necessary.

        Parameters
        ----------
        result_folder : str
            path to the folder where non rigid registration saved its result
        image_folder : str, optional
            path to the folder where the images are stored to which the
            deformations need to be applied. By default it takes the images
            used for the calculation.
        spectra_folder : str, optional
            path to the folder where the spectrum stream frames reside to which
            the deformation should be applied. If None, then no spectra are
            corrected
        """
        result_index = str(result_index)
        if result_index in self.results_paths:
            res_folder = str(Path(self.results_paths[result_index]))
        else:
            raise ValueError(f"No result found for index {result_index}")
        config_file = self.config_file_paths[result_index]
        if image_index is None:
            image_index = result_index
        if image_index in self.image_hspy_paths:
            images = hs.load(self.image_hspy_paths[result_index], lazy=True)
        # get basic info about the images
        (dataBaseName, counter, imgext, frames, skipframes, bznumber,
            stage) = _getNameCounterFrames(config_file)
        # loop over files
        im_frm_list = []
        spec_frm_list = []
        firstframe = True
        for i in range(frames):
            if i in skipframes:
                continue
            c = str(i).zfill(counter)
            imname = str(Path(
                f"{parfolder}/{imfolder}/{dataBaseName}_{c}.{imgext}"))
            image = images.get_frame(i)
            logger.info(f"Processing frame {i}: {imname}")
            if firstframe:
                defX = loadFromQ2bz(str(Path(f"{result_folder}/stage{stage}/{i}/"
                                    f"deformation_{bznumber}_0.dat.bz2")))
                defY = loadFromQ2bz(str(Path(f"{result_folder}/stage{stage}/{i}/"
                                    f"deformation_{bznumber}_1.dat.bz2")))
                firstframe = False
            else:
                defX = loadFromQ2bz(str(Path(f"{result_folder}/stage{stage}/{i}-r/"
                                    f"deformation_{bznumber}_0.dat.bz2")))
                defY = loadFromQ2bz(str(Path(f"{result_folder}/stage{stage}/{i}-r/"
                                    f"deformation_{bznumber}_1.dat.bz2")))
            w = image.width
            h = image.height
            coords = \
                np.mgrid[0:h, 0:w] + np.multiply([defY, defX], (np.max([h, w])-1))
            deformedData = ndimage.map_coordinates(image.data, coords, order=0,
                                                   mode='constant',
                                                   cval=image.data.mean())
            defImage = dio.create_new_image(deformedData, image.pixelsize,
                                            image.pixelunit, parent=image,
                                            process=("Applied non rigid "
                                                     "registration"))
            im_frm_list.append(defImage)
            if spec_list:
                logger.info("Correcting corresponding spectrum frame")
                spectra = spec_list[i]
                spectradef = dio.SpectrumStream._reshape_sparse_matrix(
                                    spectra, specstr.dimensions)
                image_stack = hs.signals.Signal2D(spectradef)
                image_stack.axes_manager[1].name = "x"
                image_stack.axes_manager[2].name = "y"
                result = image_stack.map(
                            lambda x: ndimage.map_coordinates(
                                x, coords, order=0, mode="constant"),
                            inplace=False, parallel=True)
                result.unfold()
                defspec_sp = csr_matrix(result.data.T)  # sparse matrix rep
                spec_frm_list.append(defspec_sp)
        # also do the post processing, with temmeta it's a minor thing
        logger.info("Calculating average image (undeformed)")
        resultFolder = str(Path(parfolder+f"/results_{numbering}/"))
        if not os.path.isdir(resultFolder):
            os.makedirs(resultFolder)
        # average image
        averageUndeformed = images.average()
        averageUndeformed.to_hspy(str(Path(resultFolder+"/imageUndeformed.hspy")))
        write_as_image(averageUndeformed.data,
                       str(Path(resultFolder+f"/imageUndeformed.{imgext}")))
        # average image from deformed
        logger.info("Calculating average image (deformed)")
        defstack = dio.images_to_stack(im_frm_list)
        averageDeformed = defstack.average()
        averageDeformed.to_hspy(str(Path(resultFolder+"/imageDeformed.hspy")))
        write_as_image(averageDeformed.data,
                       str(Path(resultFolder+f"/imageDeformed.{imgext}")))
        # also write out frames to individual files
        defstack.export_frames(defImagesFolder, name=dataBaseName, counter=counter)
        if spectra_folder is not None:
            # averaged spectrum
            logger.info("Calculating average spectrum (undeformed)")
            spectrumUndeformed = specstr.spectrum_map
            spectrumUndeformed.to_hspy(str(Path(
                resultFolder+"/spectrumUndeformed.hspy")))
            # averaged spectrum deformed
            logger.info("Calculating average spectrum (deformed)")
            specstr_data = dio.SpectrumStream._stack_frames(spec_frm_list)
            specstr_def = dio.SpectrumStream(specstr_data,
                                             specstr.metadata)
            # edit the number of frames
            specstr_def.metadata.data_axes["frame"]["bins"] = len(spec_frm_list)
            specstr_def.export_streamframes(defSpectraFolder,
                                            pre=dataBaseName,
                                            counter=counter)
            spectrumDeformed = specstr_def.spectrum_map
            spectrumDeformed.to_hspy(str(Path(
                    resultFolder+"/spectrumDeformed.hspy")))
            return (averageUndeformed, averageDeformed,
                    spectrumUndeformed, spectrumDeformed)
        else:
            return (averageUndeformed, averageDeformed,
                    None, None)


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


class FrameByFrame(object):
    """A pickle-able wrapper for doing a function on all frames of a stack"""
    def __init__(self, do_in_loop, stack, *args, **kwargs):
        self.func = do_in_loop
        self.stack = stack
        self.args = args
        self.kwargs = kwargs

    def __call__(self, index):
        self.func(index, self.stack, *self.args, **self.kwargs)


def export_frames(stack, output_folder=None, prefix="frame",
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
            pool.map(FrameByFrame(_save_frame_to_file, stack.data,
                                  output_folder, prefix, digits, data_format),
                     toloop)
    else:
        for i in toloop:
            logging.debug("Exporting frames sequentially")
            _save_frame_to_file(i, stack.data, output_folder, prefix, digits,
                                data_format)


def loadFromQ2bz(path):
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
    # Read magic number - only possible when bz2.open is called! Will not see
    # it in hex fiend!
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


def _getNameCounterFrames(path):
    """
    Extract relevant information from the config file for processing

    Parameters
    ----------
    path : str
        path to the .par config file

    Returns:
    tuple of (base name, number of counter digits, extension, number of frames,
              skipped frames, stoplevel, number of stages)
    """
    with open(path) as f:
        text = f.read()

    basename, counter, ext = re.findall(
        r"[\/\\]([^\/\\]+)_%0(.+)d\.([A-Za-z0-9]+)", text)[0]
    counter = int(counter)
    numframes = int(re.findall(r"numTemplates ([0-9]+)", text)[0])
    try:
        skipframes_line = re.findall(r"[^\# *]templateSkipNums (.*)", text)[0]
        skipframes = re.findall(r"([0-9]+)[^0-9]", skipframes_line)
        skipframes = list(map(int, skipframes))
    except Exception:  # when commented out, will give error
        skipframes = []
    bznumber = re.findall(r"stopLevel +([0-9]+)", text)[0].zfill(2)
    stages = int(re.findall(r"numExtraStages +([0-9]+)", text)[0])+1
    return (basename, counter, ext, numframes, skipframes, bznumber, stages)
