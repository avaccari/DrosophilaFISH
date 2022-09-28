from tkinter import filedialog
from aicsimageio import AICSImage
import napari
import numpy as np
import scipy.ndimage as ndi
import skimage.filters as ski_fil
import skimage.morphology as ski_mor
import skimage.measure as ski_mea
import seaborn as sbn
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nei
import scipy.optimize as sp_opt
import os.path as osp
import pandas as pd


# Ask user to choose a file
filename = filedialog.askopenfilenames()[0]
print("--- Starting new analysis ---")
print("File: {}".format(filename))

# Load image and extract data
original = AICSImage(filename)
data = original.get_image_data("CZYX")
pixel_sizes = original.physical_pixel_sizes
spacing = (pixel_sizes.Z, pixel_sizes.Y, pixel_sizes.X)
spacing_ratio = int(np.ceil(spacing[0] / spacing[1]))
contrast = [np.min(data), np.max(data)]
channels = original.channel_names
print("Image shape (CH, Z, Y, X): {}".format(data.shape))
print("Pixel sizes (Z, Y, X): {}".format(spacing))
print("Data type: {}".format(original.dtype))
print("Data range: {}".format(contrast))
print("Channels: {}".format(channels))

# Show original data
viewer = napari.Viewer(
    title=osp.split(filename)[1],
    ndisplay=3
)
layers = viewer.add_image(
    data,
    channel_axis=0,
    name=channels,
    colormap="gray",
    blending="additive",
    scale=spacing,
    depiction="volume",
    interpolation="nearest",
    visible=False
)

# Select the nuclei's channel
CH = 2
ch2 = data[CH, ...]

# Denoise
print("Denoising...", end="")
file_ch2_denoised = osp.splitext(filename)
file_ch2_denoised = file_ch2_denoised[0] + "-den.npy"
if osp.exists(file_ch2_denoised):
    ch2_denoised = np.load(file_ch2_denoised)
else:
    ch2_denoised = ndi.median_filter(
        ch2,
        footprint=np.ones((3, 3 * spacing_ratio, 3 * spacing_ratio))
    )
    np.save(file_ch2_denoised, ch2_denoised)
viewer.add_image(
    ch2_denoised,
    scale=spacing,
    colormap="gray",
    blending="additive",
    interpolation="nearest",
    visible=False
)
print("done!")

# Threshold
print("Thresholding...", end="")
ch2_thresholded = ch2_denoised > ski_fil.threshold_otsu(ch2_denoised)
viewer.add_image(
    ch2_thresholded,
    scale=spacing,
    opacity=0.5,
    colormap="magenta",
    blending="additive",
    interpolation="nearest",
    visible=False
)
print("done!")

# TODO: idea to try to detect center of cells. Autocorrelation: cells are about the same
#       size and autocorrelation should peak when they overlap. Or use one cell as template
#       and template matching. If we can find the centers, we can then use watershed to
#       identify individual cells.

# Evaluate connected components
print("Evaluating components...")
ch2_comp = ski_mea.label(ch2_thresholded)
ch2_masked = ch2_denoised.copy()
ch2_masked[ch2_thresholded == False] = 0
ch2_comp_props = ski_mea.regionprops(
    ch2_comp,
    intensity_image=ch2_masked
)
ch2_comp_props_df = pd.DataFrame(
    ski_mea.regionprops_table(
        ch2_comp,
        intensity_image=ch2_masked,
        properties=('label', 'area')
    )
)


# Evaluate segmented volumes statistics
ch2_min_vol, ch2_med_vol, ch2_max_vol = ch2_comp_props_df['area'].quantile([0.25, 0.5, 0.75])
ch2_mean = ch2_comp_props_df['area'].mean()
ch2_std = ch2_comp_props_df['area'].std()
print(" - Volumes quantiles: {}, {}, {}".format(ch2_min_vol, ch2_med_vol, ch2_max_vol))
ch2_med_vol = 20000  # Forcing median size by hand. TODO: find a way to evaluate based on image
print(" - Selected volume range: {} - {}".format(0.5 * ch2_med_vol, 1.5 * ch2_med_vol))
sbn.displot(
    ch2_comp_props_df,
    x='area',
    kde=True,
    rug=True
)
plt.axvline(ch2_med_vol, color="red")
plt.axvline(ch2_min_vol, color="red")
plt.axvline(ch2_max_vol, color="red")
plt.axvline(ch2_mean, color="green")
plt.axvline(ch2_mean + ch2_std, color="green")
plt.axvline(ch2_mean - ch2_std, color="green")

plt.figure()
sbn.boxplot(
    ch2_comp_props_df,
    x='area',
    notch=True,
    showcaps=False
)

# Remove large and small components
ch2_cleaned = ch2_thresholded.copy()
print(" - Removing unwanted components...", end="")
ch2_comp_props_df.loc[:, 'keep'] = False
mask = ch2_comp_props_df.query('area > 10000 & area < 25000').index
ch2_comp_props_df.loc[mask, 'keep'] = True
for _, row in ch2_comp_props_df.iterrows():
    print("{} -> {} ".format(row['label'], row['area']), end='')
    if not row['keep']:
        print('x', end='')
        ch2_cleaned[ch2_comp == row['label']] = 0
    print('')
viewer.add_image(
    ch2_cleaned,
    scale=spacing,
    opacity=0.5,
    colormap="magenta",
    blending="additive",
    interpolation="nearest",
    visible=False
)
print("done!")

# Label individual soma
print("Labeling...", end="")
ch2_labels = ski_mea.label(ch2_cleaned)
viewer.add_labels(
    ch2_labels,
    scale=spacing,
    blending="additive"
)
print("done!")

print("Mask all channels...", end="")
# Mask all channels
ch0_masked = data[0, ...].copy()
ch0_masked[ch2_cleaned == False] = 0
viewer.add_image(
    ch0_masked,
    scale=spacing,
    colormap="green",
    blending="additive",
    interpolation="nearest",
    visible=False
)

ch1_masked = data[1, ...].copy()
ch1_masked[ch2_cleaned == False] = 0
viewer.add_image(
    ch1_masked,
    scale=spacing,
    colormap="magenta",
    blending="additive",
    interpolation="nearest",
    visible=False
)

ch2_masked = data[2, ...].copy()
ch2_masked[ch2_cleaned == False] = 0
viewer.add_image(
    ch2_masked,
    scale=spacing,
    colormap="gray",
    blending="additive",
    interpolation="nearest",
    visible=False
)
print("done!")

print("Evaluate FISH...", end="")
# For each detected nucleus evaluate total brightness in each channel
# and store in table. TODO: individual fluorescent FISH should be detected.
ch2_comp_props_df.loc[:, ('ch0', 'ch1')] = 0
for idx in ch2_comp_props_df.index:
    if ch2_comp_props_df.loc[idx, 'keep']:
        mask = ch2_comp == ch2_comp_props_df['label'][idx]
        ch2_comp_props_df.loc[idx, ('ch0', 'ch1')] = data[0, mask].sum(), data[1, mask].sum()

# Plot ch0 vs ch1 for the different nuclei
plt.figure()
sbn.scatterplot(
    ch2_comp_props_df[ch2_comp_props_df['keep']],
    x='ch0',
    y='ch1'
)
top = max(ch2_comp_props_df['ch0'].max(), ch2_comp_props_df['ch1'].max())
plt.xlim((0, top))
plt.ylim((0, top))
plt.title(osp.split(filename)[1])
plt.axis('square')
print("done!")

# Show
plt.show()
napari.run()
