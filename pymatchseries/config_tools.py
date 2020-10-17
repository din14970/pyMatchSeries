import re
import os
import sys


folder, _ = os.path.split(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(folder, "default_parameters.param")

INTEGER = re.compile(r"^[0-9]+$")
FLOAT = re.compile(r"^[0-9]+\.[0-9]*$")
SCIENTIFIC = re.compile(r"^[0-9]+e-?[0-9]+$")
LIST = re.compile(r"^\{(.*)\}$")
INTEGERLISTITEMS = re.compile(r"([0-9]+)")


class config_dict(dict):
    def __init__(self, data):
        super().__init__(data)

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, key, value):
        if key not in self:
            raise KeyError(f"{key} not a valid option")
        self[key] = value

    def __getitem__(self, key):
        val = super().__getitem__(key)
        if re.match(INTEGER, val):
            return int(val)
        elif re.match(FLOAT, val) or re.match(SCIENTIFIC, val):
            return float(val)
        elif re.match(LIST, val):
            items = re.findall(INTEGERLISTITEMS, val)
            return list(map(int, items))
        else:
            return val

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError(f"{key} not a valid option")
        if isinstance(value, bool):
            value = value*1
        if isinstance(value, list):
            stringlist = " ".join(map(str, value))
            value = f"{{ {stringlist} }}"
        super().__setitem__(key, str(value))

    def save(self, path):
        write_config(self, path)


def load_config(path):
    with open(path) as f:
        config = f.read()
    options = re.findall(r"^(\w+)\s+(.+)$", config, flags=re.M)
    return config_dict(options)


def write_config(options, path):
    config = ""
    for key, value in options.items():
        config = config+str(key)+" "+str(value)+"\n"
    with open(path, "w") as f:
        f.write(config)


def create_config_file(filename, pathpattern, savedir,
                       preclevel, num_frames,
                       skipframes=[],
                       startleveloffset=2, **kwargs):
    """
    Wrapper function to create a standard config file

    Parameters
    ----------
    filename : str
        path to the config file
    pathpattern : str
        string pattern for input image files
    savedir : str
        path to output folder
    preclevel : int
        log2 of the image width and height
    num_frames : int
        number of images to process
    skipframes : list, optional
        list of indexes of frames to ignore
    startleveloffset : int, optional
        offset of the start level, lower than the `preclevel`

    Other parameters
    ----------------
    templateNamePattern : str, optional
        Same as `pathpattern`, will take precedence
    saveDirectory : str, optional
        Same as `savedir`, will take precedence
    numTemplates : int, optional
        Same as `num_frames`, will take precedence
    templateSkipNums : str, optional
        Equivalent to `skipframes`, will take precedence. Must be string.
    startLevel : int, optional
        Equivalent to preclevel - startleveloffset. Will take precedence.
    stopLevel : int, optional
        Equivalent to `preclevel`, will take precedence
    precisionLevel: int, optional
        Default is `preclevel`
    refineStartLevel: int, optional
        Default is `preclevel`-1
    refineStopLevel: int, optional
        Default is `preclevel`
    templateNumOffset : int, optional
        index of first image to process. Default is 0.
    templateNumStep : int, optional
        to skip frames at regular interval. Default is 1.
    preSmoothSigma : int, optional
        smooth images with gaussion of sigma before usage. Default is 0.
    saveRefAndTempl : bool, optional
        save both the reference and the template. Default is False.
    numExtraStages : int, optional
        number of stages. Default is 2.
    dontNormalizeInputImages : bool, optional
        do not normalize the input images. Default is False.
    enhanceContrastSaturationPercentage : float, optional
        brightness/contrast adjustment. Default is 0.15.
    normalizeMinToZero : bool, optional
        minimum intensity mapped to 0. Default is True.
    lambda : int, optional
        regularization factor used in optimization level 1. Default is 200.
    lambdaFactor : float, optional
        adjustment of reg parameter with subsequent binnings. Default is 1.
    maxGDIterations : int, optional
        max number of iterations. Default is 500.
    stopEpsilon : float, optional
        desired precision. Default is 1e-6.
    extraStagesLambdaFactor : int, optional
        adjustment of regularization with subsequent steps. Default is 0.1.
    resizeInput : bool, optional
        change size of input images. Default is False.
    dontAccumulateDeformation : bool, optional
        do not accumulate the deformations. Default is False.
    reuseStage1Results : bool, optional
        Default is 1
    useMedianAsNewTarget : bool, optional
        Use median versus mean. Default is 1.
    calcInverseDeformation : bool, optional
        Default is 0
    skipStage1 : bool, optional
        Default is 0
    saveNamedDeformedTemplates : bool, optional
        Default is 1
    saveNamedDeformedTemplatesUsingNearestNeighborInterpolation : bool,optional
        Default is 1
    saveNamedDeformedTemplatesExtendedWithMean: bool, optional
        Default is 1
    saveDeformedTemplates : bool, optional
        Default is 1
    saveNamedDeformedDMXTemplatesAsDMX : bool, optional
        Default is 1
    """
    try:
        config_dict = load_config(DEFAULT_CONFIG_PATH)
    except Exception as e:
        print(f"Something went wrong reading the default config file: {e}")
        sys.exit()
    config_dict["templateNamePattern"] = pathpattern
    config_dict["saveDirectory"] = savedir
    config_dict["numTemplates"] = num_frames
    skipframes = " ".join(map(str, skipframes))
    config_dict["templateSkipNums"] = f"{{ {skipframes} }}"
    config_dict["startLevel"] = preclevel - startleveloffset
    config_dict["stopLevel"] = preclevel
    config_dict["precisionLevel"] = preclevel
    config_dict["refineStartLevel"] = preclevel-1
    config_dict["refineStopLevel"] = preclevel
    for key, value in kwargs.items():
        if isinstance(value, bool):
            value = value*1
        config_dict[key] = value
    config_dict.save(filename)
    print(f"Created config file in {os.path.abspath(filename)}")
