import json
import os

from os_utils import build_path


class Metadata:
    def __init__(self, filename_root, suf="-meta.json", out_dir=None):
        self.root_dir = build_path(filename_root, out_dir=out_dir)
        self.file = build_path(filename_root, suf, out_dir=out_dir)

    def load_metadata(self, sec_data=None):
        try:
            with open(self.file) as meta_file:
                meta_data = json.load(meta_file)
                return meta_data if sec_data is None else meta_data[sec_data]
        except FileNotFoundError:
            print("WARNING: Metadata file not found!")
            return None
        except json.decoder.JSONDecodeError:
            print("WARNING: Metadata file is not in the correct format!")
            return None
        except KeyError:
            print(f"WARNING: Section {sec_data} not found in metadata file!")
            return {}

    def save_metadata(self, sec_name=None, sec_data=None):
        try:
            if not os.path.isdir(self.root_dir):
                os.makedirs(self.root_dir)
            with open(self.file, "w") as meta_file:
                if meta_data is None:
                    meta_data = {sec_name: sec_data}
                else:
                    meta_data[sec_name] = sec_data
                json.dump(meta_data, meta_file)
        except OSError:
            print("WARNING: Issues opening the metadata file!")
