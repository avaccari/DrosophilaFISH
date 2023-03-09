import os
import numpy as np
import lzma


def build_path(filename_root, suffix=None):
    dir, file = os.path.split(filename_root)
    file_name, _ = os.path.splitext(file)
    root_dir = os.path.join(dir, file_name)
    return root_dir if suffix is None else os.path.join(root_dir, file_name + suffix)


def save(file, array):
    print("saving data...", end='', flush=True)
    with lzma.open(f"{file}.lzma", "wb") as f:
        np.save(f, array)


def load(file):
    print("loading data...", end='', flush=True)
    with lzma.open(f"{file}.lzma", "rb") as f:
        return np.load(f)
