import os
import numpy as np
import lzma

import tifffile


def build_path(filename_root, suffix=None):
    dir, file = os.path.split(filename_root)
    file_name, _ = os.path.splitext(file)
    root_dir = os.path.join(dir, file_name)
    return root_dir if suffix is None else os.path.join(root_dir, file_name + suffix)


def save(file, array):
    print("saving data... ", end="", flush=True)
    with lzma.open(f"{file}.lzma", "wb") as f:
        np.save(f, array)


def load(file):
    if not os.path.exists(f"{file}.lzma"):
        raise FileNotFoundError

    print("loading data... ", end="", flush=True)
    with lzma.open(f"{file}.lzma", "rb") as f:
        return np.load(f)


def store_output(function, filename_root=None, ch_id=None, suffix="", args=None):
    if args is None:
        args = {}
    if filename_root is None:
        output = function(**args)
    elif ch_id is None:
        raise ValueError(
            "A ch_id should be provided to identify the channel. Segmentation was not evaluated."
        )
    else:
        file = build_path(filename_root, f"-{ch_id}-{suffix}.npy")
        try:
            output = load(file)
        except (FileNotFoundError, ValueError):
            output = function(**args)
            try:
                root_dir = build_path(filename_root)
                if not os.path.isdir(root_dir):
                    os.makedirs(root_dir)
                save(file, output)
            except Exception:
                print("WARNING: error saving the file.")
    print("done!")
    return output


def write_to_tif(array, filename_root=None, ch_id=None, suffix=""):
    if ch_id is None:
        raise ValueError(
            "A ch_id should be provided to identify the channel. Image is not written."
        )
    elif filename_root is None:
        raise ValueError(
            "A filename_root should be provided to identify the file. Image is not written."
        )
    else:
        print(f"Writing {ch_id} as image...", end="", flush=True)
        file = build_path(filename_root, f"-{ch_id}-{suffix}.tif")
        try:
            root_dir = build_path(filename_root)
            if not os.path.isdir(root_dir):
                os.makedirs(root_dir)
            tifffile.imwrite(
                file,
                array,
                imagej=True,
                compression="zlib",
                compressionargs={"level": 9},
            )
            print("done!")
        except Exception:
            print("WARNING: error saving the file.")
