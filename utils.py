from tkinter import filedialog
from aicsimageio import AICSImage
import numpy as np
import scipy.ndimage as ndi
import os.path as osp
from glob import glob
import skimage.exposure as ski_exp
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt


# Denoise specific channel for an entire folder and save results
def denoise_folder(folder=None, channel=None, footprint_dim=None, base_dim=3):
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
            footprint = np.ones((base_dim, base_dim * spacing_ratio, base_dim * spacing_ratio))
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
            print("- Contrast stretching channel {}...".format(ch), end="")
            ch_data = ski_exp.rescale_intensity(ch_data)
            print("done!")
            print("- Denoising channel {}...".format(ch) , end="")
            file_denoised = osp.splitext(f)
            file_denoised = file_denoised[0] + "-{}-den-{}-{}.npy".format(data.size, ch, footprint.shape)
            if not osp.exists(file_denoised):
                denoised = ndi.median_filter(
                    ch_data,
                    footprint=footprint
                )
                np.save(file_denoised, denoised)
            print("done!")


# Combined detection datasets and plot
def plot_combined_detections(folder=None, file_mask='*', x_lbl='Cam1-T2', y_lbl='Cam2-T1'):
    if folder is None:
        folder = filedialog.askdirectory()
    try:
        files = glob(folder + '/' + file_mask)
    except:
        print("folder {} not available!".format(folder))
        quit()

    fig, ax = plt.subplots()
    labels=[]
    for f in files:
        df = pd.read_json(f)
        sbn.scatterplot(
            df[df['keep']],
            x=x_lbl + '_cnt',
            y=y_lbl + '_cnt',
            ax=ax
        )
        labels.append(osp.splitext(osp.split(f)[1])[0])
    top = max(df[x_lbl + '_cnt'].max(),
              df[y_lbl + '_cnt'].max())
    plt.xlim((0, top))
    plt.ylim((0, top))
    plt.title('647 (y-axis) vs. 568 (x-axis)')
    plt.axis('square')
    plt.legend(labels=labels, loc='upper left', borderaxespad=0, fontsize=8)

# denoise_folder(folder="/Volumes/GoogleDrive/Shared drives/MarkD/FISH analysis/FISH dataset (2x exp) Sep 2022/",
#                footprint_dim=(3, 3, 3))
# denoise_folder(folder="/Volumes/GoogleDrive/Shared drives/MarkD/FISH analysis/FISH dataset (2x exp) Sep 2022/",
#                footprint_dim=(6, 6, 6))
# denoise_folder(folder="/Volumes/GoogleDrive/Shared drives/MarkD/FISH analysis/FISH dataset (2x exp) Sep 2022/",
#                channel=[0],
#                base_dim=3)
# denoise_folder(folder="/Volumes/GoogleDrive/Shared drives/MarkD/FISH analysis/FISH dataset (2x exp) Sep 2022/",
#                channel=[0],
#                base_dim=6)