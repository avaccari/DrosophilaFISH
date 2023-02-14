import logging
import sys
import numpy as np
from czi import Czi
from channel import Channel


class Image:
    def __init__(self, filename):
        self._logger = self._setup_logger(__name__)
        self.filename = filename
        self.scaling = None
        self.scale_ratio = None
        self.data = None
        self.metadata = None
        self.channels_meta = None
        self.type_meta = None

    def _setup_logger(self, name):
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        # Add console handler
        formatter = logging.Formatter(
            "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
        )
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def load_image(self):
        with Czi(self.filename) as czi:
            self.metadata = czi.get_metadata()
            self.data = czi.get_data()

            self.scaling = self.metadata["scaling_z_y_x"]
            self.scale_ratio = np.asarray(self.scaling) / min(self.scaling)
            self.channels_meta = self.metadata["channels"]
            self.type_meta = self.metadata["image_type"]
            

            # NOTE: Temporary removed until fully implemented in main.py
            # self.channels = []
            # for ch in range(len(self.channels_meta)):
            #     self.channels.append(
            #         Channel(self.data[ch], self.scaling, self.channels_meta[ch])
            #     )

    def get_data(self):
        return self.data

    def get_metadata(self):
        return self.metadata
