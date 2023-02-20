import logging
import sys
from czifile import CziFile
import xmltodict


class Czi:
    def __init__(self, filename):
        self._logger = self._setup_logger(__name__)
        self.filename = filename
        self.czi = None
        try:
            self.czi = CziFile(filename)
        except FileNotFoundError:
            self._logger.warning(f"File {filename} not found!")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.czi.close()

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
        czi_meta = xmltodict.parse(self.czi.metadata())
        scaling = czi_meta["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
        d = {d["@Id"]: float(d["Value"]) for d in scaling}
        channels = czi_meta["ImageDocument"]["Metadata"]["Information"]["Image"][
            "Dimensions"
        ]["Channels"]["Channel"]
        ch = {
            c: {
                "Id": channels[c]["@Id"],
                "Name": channels[c]["@Name"],
                "Wavelength": channels[c]["EmissionWavelength"],
            }
            for c in range(len(channels))
        }
        return {
            "scaling_z_y_x": (d["Z"], d["Y"], d["X"]),
            "image_type": {
                "bit_depth": czi_meta["ImageDocument"]["Metadata"]["Information"][
                    "Image"
                ]["ComponentBitCount"],
                "type": czi_meta["ImageDocument"]["Metadata"]["Information"]["Image"][
                    "PixelType"
                ],
            },
            "channels": ch,
        }

    def get_data(self):
        self._logger.debug("Reading data")

        return self.czi.asarray().squeeze()
