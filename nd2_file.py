import logging
import sys
import numpy as np
from nd2 import ND2File


class Nd2:
    def __init__(self, filename):
        self._logger = self._setup_logger(__name__)
        self.filename = filename
        self.nd2 = None
        try:
            self.nd2 = ND2File(filename)
        except FileNotFoundError:
            self._logger.warning(f"File {filename} not found!")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.nd2.close()

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
        sizes = self.nd2.sizes
        size_z = int(sizes.get("Z", 1))
        size_y = int(sizes["Y"])
        size_x = int(sizes["X"])
        # ND2 voxel sizes are in µm, CZI ones in m: convert to match
        voxel = self.nd2.voxel_size()
        scale = {"Z": voxel.z * 1e-6, "Y": voxel.y * 1e-6, "X": voxel.x * 1e-6}
        channels_no = int(sizes.get("C", 1))
        channels = self.nd2.metadata.channels or []
        ch = {
            c: {
                "Id": f"Channel:{c}",
                "Name": channels[c].channel.name,
                "Wavelength": self._get_wavelength(channels[c]),
            }
            for c in range(len(channels))
        }
        attributes = self.nd2.attributes
        return {
            "size_z_y_x": (size_z, size_y, size_x),
            "scaling_z_y_x": (scale["Z"], scale["Y"], scale["X"]),
            "image_type": {
                "bit_depth": attributes.bitsPerComponentSignificant,
                "type": str(self.nd2.dtype),
            },
            "channels_no": channels_no,
            "channels": ch,
        }

    def _get_wavelength(self, channel):
        # Prefer the emission wavelength (as in CZI), fall back to excitation
        emission = channel.channel.emissionLambdaNm
        if emission:
            return emission
        excitation = channel.channel.excitationLambdaNm
        if excitation:
            return excitation
        return 0.0

    def get_data(self):
        self._logger.debug("Reading data")
        data = self.nd2.asarray()

        # Reorder axes as (C, Z, Y, X) to match the CZI layout
        axes = list(self.nd2.sizes.keys())
        if "C" in axes and "Z" in axes and axes.index("C") > axes.index("Z"):
            data = np.moveaxis(data, axes.index("C"), axes.index("Z"))

        return data.squeeze()
