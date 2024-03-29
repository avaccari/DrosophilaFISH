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
        size_z = int(
            czi_meta["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeZ"]
        )
        size_y = int(
            czi_meta["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeY"]
        )
        size_x = int(
            czi_meta["ImageDocument"]["Metadata"]["Information"]["Image"]["SizeX"]
        )
        scaling = czi_meta["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"]
        scale = {entry["@Id"]: float(entry["Value"]) for entry in scaling}
        channels_no = czi_meta["ImageDocument"]["Metadata"]["Information"]["Image"][
            "SizeC"
        ]
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
            "size_z_y_x": (size_z, size_y, size_x),
            "scaling_z_y_x": (scale["Z"], scale["Y"], scale["X"]),
            "image_type": {
                "bit_depth": czi_meta["ImageDocument"]["Metadata"]["Information"][
                    "Image"
                ]["ComponentBitCount"],
                "type": czi_meta["ImageDocument"]["Metadata"]["Information"]["Image"][
                    "PixelType"
                ],
            },
            "channels_no": channels_no,
            "channels": ch,
        }

    def get_data(self):
        self._logger.debug("Reading data")

        return self.czi.asarray().squeeze()
