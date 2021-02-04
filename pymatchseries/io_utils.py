from PIL import Image
from pathlib import Path
import concurrent.futures as cf
import logging
import os
import bz2
import numpy as np


def _save_frame_to_file(i, data, folder, prefix, digits, data_format="tiff"):
    """
    Helper function for multithreading, saving frame i of stack
    """
    c = str(i).zfill(digits)
    fp = str(Path(f"{folder}/{prefix}_{c}.{data_format}"))
    frm = data[i]
    try:
        # in the case of dask
        frm = frm.compute()
    except Exception:
        pass
    # in case of a weird datatype
    if not (frm.dtype == np.uint16 or frm.dtype == np.uint8):
        frm = (frm - frm.min()) / (frm.max() - frm.min()) * (2 ** 16 - 1)
        frm = np.uint16(frm)
    img = Image.fromarray(frm)
    img.save(fp)
    logging.debug(f"Wrote out image {prefix}_{c}")


class _FrameByFrame(object):
    """A pickle-able wrapper for doing a function on all frames of a stack"""

    def __init__(self, do_in_loop, stack, *args, **kwargs):
        self.func = do_in_loop
        self.stack = stack
        self.args = args
        self.kwargs = kwargs

    def __call__(self, index):
        self.func(index, self.stack, *self.args, **self.kwargs)


def export_frames(
    data, folder, prefix, digits, frames=None, multithreading=True, data_format="tiff"
):
    """
    Export a 3D data array as individual images as required by matchseries
    """
    if frames is None:
        toloop = range(data.shape[0])
    elif isinstance(frames, list):
        toloop = frames
    else:
        raise TypeError("Argument frames must be a list")
    if multithreading:
        with cf.ThreadPoolExecutor() as pool:
            logging.debug("Starting in parallel mode to export")
            pool.map(
                _FrameByFrame(
                    _save_frame_to_file, data, folder, prefix, digits, data_format
                ),
                toloop,
            )
    else:
        for i in toloop:
            logging.debug("Exporting frames sequentially")
            _save_frame_to_file(i, data, folder, prefix, digits, data_format)


def _loadFromQ2bz(path):
    """
    Opens a bz2 or q2bz file and returns an "image" = the deformations
    """
    filename, file_extension = os.path.splitext(path)
    # bz2 compresses only a single file
    if file_extension == ".q2bz" or file_extension == ".bz2":
        # read binary mode, r+b would be to also write
        fid = bz2.open(path, "rb")
    else:
        fid = open(path, "rb")  # read binary mode, r+b would be to also write
    # rstrip removes trailing zeros
    binaryline = fid.readline()  # will look like "b"P9\n""
    line = binaryline.rstrip().decode("ascii")  # this will look like "P9"
    if line[0] != "P":
        raise ValueError("Invalid array header, doesn't start with 'P'")
    if line[1] == "9":
        dtype = np.float64
    elif line[1] == "8":
        dtype = np.float32
    else:
        dtype = None

    if not dtype:
        raise NotImplementedError(
            f"Invalid data type ({line[1]}), only float and "
            "double are supported currently"
        )
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
        if c == b"\n":
            break
        max = max + str(int(c))

    max = int(max)
    # Read image to vector
    x = np.frombuffer(fid.read(), dtype)
    img = x.reshape(height, width)
    return img


def overwrite_file(fname):
    """If file exists 'fname', ask for overwriting and return True or False,
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


def overwrite_dir(fname):
    """If dir exists 'fname', ask for overwriting and return True or False,
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
