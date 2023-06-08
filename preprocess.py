import numpy as np
import scipy.ndimage as sci_ndi
import skimage.exposure as ski_exp
import skimage.filters.rank as ski_fil_ran
import skimage.morphology as ski_mor
from colorama import Fore, Style

import os_utils


def eval_stats(data, mask=None):
    masked = data[mask] if mask is not None else data
    return masked.min(), np.median(masked), masked.max()


def contrast_stretch(data, in_range="image", mask=None, ch_id=None):
    print(
        f"Contrast stretching {Fore.GREEN}{Style.BRIGHT}{ch_id}{Style.RESET_ALL}... ",
        end="",
        flush=True,
    )
    input_min, input_median, input_max = eval_stats(data, mask)
    data = ski_exp.rescale_intensity(data, in_range)
    scaled_min, scaled_median, scaled_max = eval_stats(data, mask)
    print(
        f"{Style.BRIGHT}[{input_min}, {input_median}, {input_max}] => {in_range} => [{scaled_min}, {scaled_median}, {scaled_max}]{Style.RESET_ALL}... done!"
    )
    return data


def filter(
    data,
    type="median",
    footprint=None,
    sigma=(2, 10, 10),
    mode="nearest",
    cval=0,
    filename_root=None,
    ch_id=None,
):
    filtered = data.copy()

    if footprint is None:
        footprint = ski_mor.ball(1)
    footprint_dim = footprint.shape

    if type == "closing":
        fun = ski_mor.closing
        suffix = f"clos-{footprint_dim}"
    elif type == "gaussian":
        fun = sci_ndi.gaussian_filter
        suffix = f"den-gaus-{tuple(np.round(sigma, decimals=2))}"
    elif type == "maximum":
        fun = ski_fil_ran.maximum
        suffix = f"max-{footprint_dim}"
    elif type == "median":
        fun = sci_ndi.median_filter
        suffix = f"den-med-{footprint_dim}"
    else:
        raise ValueError("WARNING: the specified filtering mode is not available.")

    print(
        f"Applying {Fore.BLUE}{type}{Style.RESET_ALL} filter to {Fore.GREEN}{Style.BRIGHT}{ch_id}{Style.RESET_ALL}... ",
        end="",
        flush=True,
    )

    filtered = os_utils.store_output(
        fun,
        filename_root=filename_root,
        ch_id=ch_id,
        suffix=suffix,
        data=data,
        footprint=footprint,
        sigma=sigma,
        mode=mode,
        cval=cval,
    )

    print("done!")

    return filtered


def remove_floor(data, sigma=(20, 100, 100), filename_root=None, ch_id=None, mask=None):
    print(f"Removing floor from {Fore.GREEN}{Style.BRIGHT}{ch_id}{Style.RESET_ALL}:")
    input_min, input_median, input_max = eval_stats(data, mask=mask)
    noise_floor = filter(
        data, type="gaussian", sigma=sigma, filename_root=filename_root, ch_id=ch_id
    )
    defloored = data.astype("int16") - noise_floor.astype("int16")
    defloored = np.maximum(defloored, 0).astype("uint8")
    defloored_min, defloored_median, defloored_max = eval_stats(defloored, mask=mask)
    print(
        f"{Style.BRIGHT}[{input_min}, {input_median}, {input_max}] => [{defloored_min}, {defloored_median}, {defloored_max}]{Style.RESET_ALL}"
    )

    return defloored
