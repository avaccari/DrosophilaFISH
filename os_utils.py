import os


def build_path(filename_root, suffix=None):
    dir, file = os.path.split(filename_root)
    file_name, _ = os.path.splitext(file)
    root_dir = os.path.join(dir, file_name)
    return root_dir if suffix is None else os.path.join(root_dir, file_name + suffix)
