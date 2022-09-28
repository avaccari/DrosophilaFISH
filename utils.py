from tkinter import filedialog
from aicsimageio import AICSImage
import numpy as np
import scipy.ndimage as ndi
import os.path as osp
from glob import glob


# Denoise specific channel for an entire folder and save results
def denoise_folder(folder=None, channel=None, footprint_dim=None):
    if folder is None:
        folder = filedialog.askdirectory()
    try:
        files = glob(folder + "/*.czi")
    except:
        print("folder {} not available!".format(folder))
        quit()
    for it in range(len(files)):
        f = files[it]
        print("File ({}/{}): {}".format(it + 1, len(files), f))
        try:
            original = AICSImage(f)
        except:
            print("Error opening {}".format(f))
            quit()
        data = original.get_image_data("CZYX")
        pixel_sizes = original.physical_pixel_sizes
        spacing = (pixel_sizes.Z, pixel_sizes.Y, pixel_sizes.X)
        spacing_ratio = int(np.ceil(spacing[0] / spacing[1]))
        contrast = [np.min(data), np.max(data)]
        channels = original.channel_names
        print("- Image shape (CH, Z, Y, X): {}".format(data.shape))
        print("- Pixel sizes (Z, Y, X): {}".format(spacing))
        print("- Data type: {}".format(original.dtype))
        print("- Data range: {}".format(contrast))
        print("- Channels: {}".format(channels))
        if footprint_dim is None:
            footprint = np.ones((3, 3 * spacing_ratio, 3 * spacing_ratio))
        else:
            footprint = np.ones(footprint_dim)
        print("- Footprint for median: {}".format(footprint.shape))
        if channel is None:
            chan = range(len(channels))
        else:
            chan = channel
        for ch in chan:
            try:
                ch_data = data[ch, ...]
            except:
                print("Channel {} not available!".format(ch))
                quit()
            print("- Denoising channel {}...".format(ch) , end="")
            file_denoised = osp.splitext(f)
            file_denoised = file_denoised[0] + "-den-{}-{}.npy".format(ch, footprint.shape)
            if not osp.exists(file_denoised):
                denoised = ndi.median_filter(
                    ch_data,
                    footprint=footprint
                )
                np.save(file_denoised, denoised)
            print("done!")

denoise_folder(footprint_dim=(3, 3, 3))