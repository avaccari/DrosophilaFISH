import logging
import sys
import numpy as np


class Npy:
    def __init__(self, filename):
        self._logger = self._setup_logger(__name__)
        self.filename = filename
        self.npy = None
        try:
            self.npy = np.load(filename, mmap_mode="r")
        except FileNotFoundError:
            self._logger.warning(f"File {filename} not found!")
        except Exception:
            raise ValueError("The file is not a valid .npy file.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Memory-mapped arrays are closed when no longer referenced
        self.npy = None

    def _setup_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

        return logger

    def get_metadata(self):
        self._logger.debug("Reading metadata")
        shape = self.npy.shape
        channels_no = shape[0] if len(shape) == 4 else 1
        size_z, size_y, size_x = shape[1:] if len(shape) == 4 else shape
        # NPY files carry no physical scaling or channel information
        return {
            "size_z_y_x": (size_z, size_y, size_x),
            "scaling_z_y_x": (1.0, 1.0, 1.0),
            "image_type": {
                "bit_depth": self.npy.dtype.itemsize * 8,
                "type": str(self.npy.dtype),
            },
            "channels_no": channels_no,
            "channels": {},
        }

    def get_data(self):
        self._logger.debug("Reading data")

        return np.array(self.npy)
