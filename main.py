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
viewer.add_image(
    data,
    channel_axis=0,
    name=channels,
    colormap="green",
    blending="additive",
    scale=spacing,
    depiction="volume",
    interpolation="nearest",
    visible=False
)

# Denoise
footprint_dim = ((3, 3, 3),
                 (3, 3, 3),
                 (3, 3, 3))
                 # (3, 3 * spacing_ratio, 3 * spacing_ratio))
denoised = np.empty_like(data)
print("Denoising...")
for ch in range(len(channels)):
    print("- Denoising channel {}...".format(ch), end="")
    file = osp.splitext(filename)
    file = file[0] + "-den-{}-{}.npy".format(ch, footprint_dim[ch])
    if osp.exists(file):
        denoised[ch, ...] = np.load(file)
    else:
        denoised[ch, ...] = ndi.median_filter(
            data[ch, ...],
            footprint=np.ones(footprint_dim[ch])
        )
        np.save(file, denoised[ch, ...])
    print("done!")
names = [c + '-den' for c in channels]
viewer.add_image(
    denoised,
    channel_axis=0,
    name=names,
    colormap="magenta",
    blending="additive",
    scale=spacing,
    interpolation="nearest",
    visible=False
)

# Threshold to identify the nuclei
print("Thresholding...", end="")
NUCLEI_CH = 2
nuclei_mask = denoised[NUCLEI_CH] > ski_fil.threshold_otsu(denoised)
viewer.add_image(
    nuclei_mask,
    scale=spacing,
    colormap="blue",
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
nuclei_comp = ski_mea.label(nuclei_mask)
nuclei_props_df = pd.DataFrame(
    ski_mea.regionprops_table(
        nuclei_comp,
        properties=('label', 'area')
    )
)

# Evaluate and store components properties for all channels
for ch in range(len(channels)):
    comp_props_df = pd.DataFrame(
        ski_mea.regionprops_table(
            nuclei_comp,
            intensity_image=data[ch, ...],
            properties=('label', 'intensity_mean')
        )
    )
    nuclei_props_df = nuclei_props_df.merge(comp_props_df, on='label')
    nuclei_props_df = nuclei_props_df.rename(columns={'intensity_mean': 'intensity_mean_ch{}'.format(ch)})

# Evaluate nuclei's volume statistics
nuclei_min_vol, nuclei_med_vol, nuclei_max_vol = nuclei_props_df['area'].quantile([0.25, 0.5, 0.75])
nuclei_mean = nuclei_props_df['area'].mean()
nuclei_std = nuclei_props_df['area'].std()
print(" - Volumes quantiles: {}, {}, {}".format(nuclei_min_vol, nuclei_med_vol, nuclei_max_vol))
nuclei_med_vol = 20000  # Forcing median size by hand. TODO: find a way to evaluate based on image
print(" - Selected volume range: {} - {}".format(0.5 * nuclei_med_vol, 1.5 * nuclei_med_vol))
sbn.displot(
    nuclei_props_df,
    x='area',
    kde=True,
    rug=True
)
plt.axvline(nuclei_med_vol, color="red")
plt.axvline(nuclei_min_vol, color="red")
plt.axvline(nuclei_max_vol, color="red")
plt.axvline(nuclei_mean, color="green")
plt.axvline(nuclei_mean + nuclei_std, color="green")
plt.axvline(nuclei_mean - nuclei_std, color="green")

plt.figure()
sbn.boxplot(
    nuclei_props_df,
    x='area',
    notch=True,
    showcaps=False
)

# Remove large and small components
nuclei_mask_cleaned = nuclei_mask.copy()
print(" - Removing unwanted components...", end="")
nuclei_props_df.loc[:, 'keep'] = False
mask = nuclei_props_df.query('area > 10000 & area < 25000').index
nuclei_props_df.loc[mask, 'keep'] = True
for _, row in nuclei_props_df.iterrows():
    # print("{} -> {} ".format(row['label'], row['area']), end='')
    if not row['keep']:
        # print('x', end='')
        nuclei_mask_cleaned[nuclei_comp == row['label']] = False
    # print('')

viewer.add_image(
    nuclei_mask_cleaned,
    scale=spacing,
    colormap="blue",
    blending="additive",
    interpolation="nearest",
    visible=False
)
print("done!")

# Label individual soma
print("Labeling...", end="")
nuclei_labels = ski_mea.label(nuclei_mask_cleaned)
viewer.add_labels(
    nuclei_labels,
    scale=spacing,
    blending="additive",
    visible=False
)
print("done!")

# Mask all channels
print("Masking...")
masked = np.empty_like(data)
for ch in range(len(channels)):
    print("- Masking channel {}...".format(ch), end="")
    masked[ch, ...] = denoised[ch, ...].copy()
    masked[ch, nuclei_mask_cleaned == False] = 0
    print("done!")

names = [c + '-msk' for c in channels]
viewer.add_image(
    masked,
    channel_axis=0,
    name=names,
    scale=spacing,
    colormap=["green", "magenta", "gray"],
    blending="additive",
    interpolation="nearest",
    visible=False
)

print("Evaluate FISH...", end="")
plt.figure()
sbn.scatterplot(
    nuclei_props_df[nuclei_props_df['keep']],
    x='intensity_mean_ch0',
    y='intensity_mean_ch1',
    size='intensity_mean_ch2',
    hue='label'
)
top = max(nuclei_props_df['intensity_mean_ch0'].max(), nuclei_props_df['intensity_mean_ch1'].max())
plt.xlim((0, top))
plt.ylim((0, top))
plt.title(osp.split(filename)[1])
plt.axis('square')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
print("done!")

# Show
plt.show()
napari.run()
