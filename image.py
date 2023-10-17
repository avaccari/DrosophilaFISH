import logging
import sys
import numpy as np
from czi import Czi
from colorama import Fore, Style

# from channel import Channel


class Image:
    def __init__(
        self,
        filename,
        resolution=None,
        metadata_only=False,
        required_channels=None,
        nuclei_ch=0,
        cytoplasm_ch=3,
    ):
        self.filename = filename
        self.resolution = resolution
        self.metadata = None
        self.channels_no = None
        self.channels_meta = None
        self.scaling = None
        self.scale_ratio = None
        self.size = None
        self.type_meta = None

        self._logger = self._setup_logger(__name__)
        self._get_metadata()
        self._show_metadata()

        if not metadata_only:
            if required_channels != self.channels_no:
                raise ValueError(
                    f"Number of required channels ({required_channels}) does not match the number of channels in the image ({self.channels_no})."
                )
            self.ch_dict = self._get_ch_dict(required_channels, nuclei_ch, cytoplasm_ch)

            self.load_image()

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
        if self.filename.endswith(".czi"):
            with Czi(self.filename) as czi:
                self.data = czi.get_data()
        if self.filename.endswith(".npy"):
            try:
                self.data = np.load(self.filename)
            except Exception:
                raise ValueError("The file is not a valid .npy file.")

        # Check the number of channels and if only 1, make it 4D
        if len(self.data.shape) == 3:
            self.data = np.expand_dims(self.data, axis=0)

        contrast = [
            [np.min(self.data[ch]), np.max(self.data[ch])]
            for ch in range(self.data.shape[0])
        ]
        print(
            f"{Style.BRIGHT}{Fore.BLUE}############   Image data:   ############{Style.RESET_ALL}"
        )
        print(f"{Style.BRIGHT}Data ranges:{Style.RESET_ALL}")
        for (c, n), r in zip(
            [(k, v) for k, v in self.ch_dict.items() if isinstance(k, int)], contrast
        ):
            print(f"  {c}: {n:9} => {r}")
        print(
            f"{Style.BRIGHT}{Fore.BLUE}#########################################{Style.RESET_ALL}"
        )

        # NOTE: Temporary removed until fully implemented in main.py
        # self.channels = []
        # for ch in range(self.channels_no):
        #     self.channels.append(
        #         Channel(self.data[ch], self.scaling, self.channels_meta[ch])
        #     )

    def _get_metadata(self):
        if self.filename.endswith(".czi"):
            with Czi(self.filename) as czi:
                self.metadata = czi.get_metadata()
                self.channels_meta = self.metadata["channels"]
                self.channels_no = int(self.metadata["channels_no"])
                # self.channels_no = len(self.channels_meta)
                self.size = self.metadata["size_z_y_x"]
                if self.resolution is None:
                    self.scaling = self.metadata["scaling_z_y_x"]
                else:
                    self.scaling = self.resolution
                self.scale_ratio = max(np.asarray(self.scaling) / min(self.scaling))
                self.type_meta = self.metadata["image_type"]
        if self.filename.endswith(".npy"):
            self.metadata = {}
            self.channels_meta = {}
            img = np.load(self.filename, mmap_mode="r")
            shape = img.shape
            self.channels_no = shape[0] if len(shape) == 4 else 1
            self.size = shape[1:] if len(shape) == 4 else shape
            if self.resolution is None:
                self.scaling = (
                    1.0,
                    1.0,
                    1.0,
                )
            else:
                self.scaling = self.resolution
            self.scale_ratio = max(np.asarray(self.scaling) / min(self.scaling))
            self.type_meta = {"type:": img.dtype}

    def _show_metadata(self):
        print(
            f"{Style.BRIGHT}{Fore.BLUE}############ Image metadata: ############{Style.RESET_ALL}"
        )
        print(f"{Style.BRIGHT}Image shape (Z, Y, X):{Style.RESET_ALL}")
        print(f"  {self.channels_no} channels => {self.size}")
        print(f"{Style.BRIGHT}Channels:{Style.RESET_ALL}")
        [print(f"  {k} <=> {v}") for k, v in self.channels_meta.items()]
        print(f"{Style.BRIGHT}Pixel sizes (Z, Y, X):{Style.RESET_ALL}")
        print(f"  {self.scaling}")
        print(f"{Style.BRIGHT}Spacing ratio (Z / X or Y):{Style.RESET_ALL}")
        print(f"  {self.scale_ratio}")
        print(f"{Style.BRIGHT}Data type:{Style.RESET_ALL}")
        for k, v in self.type_meta.items():
            print(f"  {k} => {v}")
        print(
            f"{Style.BRIGHT}{Fore.BLUE}#########################################{Style.RESET_ALL}"
        )

    @staticmethod
    def _get_ch_dict(required_channels, nuclei_ch, cytoplasm_ch):
        ch_dict = {}
        if required_channels == 4:
            if cytoplasm_ch is None:
                raise ValueError(
                    f"Configuration of channels ({required_channels}) and cytoplasm_ch ({cytoplasm_ch}) not supported."
                )
            if nuclei_ch == 1:
                ch_dict["Nuclei"] = 1
                ch_dict["Cytoplasm"] = 3
                ch_dict["others"] = [1, 3]
                ch_dict["fish"] = [0, 2]
                ch_dict[0] = "FISH_647"
                ch_dict[1] = "Nuclei"
                ch_dict[2] = "FISH_568"
                ch_dict[3] = "Cytoplasm"
                ch_dict["colormaps"] = ["bop blue", "green", "bop orange", "gray"]
            else:
                ch_dict["Nuclei"] = 0
                ch_dict["Cytoplasm"] = 3
                ch_dict["others"] = [0, 3]
                ch_dict["fish"] = [1, 2]
                ch_dict[0] = "Nuclei"
                ch_dict[1] = "FISH_647"
                ch_dict[2] = "FISH_568"
                ch_dict[3] = "Cytoplasm"
                ch_dict["colormaps"] = ["green", "bop blue", "bop orange", "gray"]
        elif required_channels == 3:
            if nuclei_ch == 1:
                raise ValueError(
                    f"Configuration of channels ({required_channels}) and nuclei_ch ({nuclei_ch}) not supported."
                )
            if cytoplasm_ch is None:
                ch_dict["Nuclei"] = 0
                ch_dict["others"] = [0]
                ch_dict["fish"] = [1, 2]
                ch_dict[0] = "Nuclei"
                ch_dict[1] = "FISH_647"
                ch_dict[2] = "FISH_568"
                ch_dict["colormaps"] = ["green", "bop blue", "bop orange"]
            else:
                ch_dict["Nuclei"] = 0
                ch_dict["Cytoplasm"] = 2
                ch_dict["others"] = [0, 2]
                ch_dict["fish"] = [1]
                ch_dict[0] = "Nuclei"
                ch_dict[1] = "FISH_568"
                ch_dict[2] = "Cytoplasm"
                ch_dict["colormaps"] = ["green", "bop orange", "gray"]
        elif required_channels == 1:
            ch_dict["Nuclei"] = 0
            ch_dict["others"] = [0]
            ch_dict[0] = "Nuclei"
            ch_dict["colormaps"] = ["green"]
        else:
            raise ValueError(f"Number of channels ({required_channels}) not supported.")

        return ch_dict
