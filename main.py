from tkinter import filedialog

import skimage
from aicsimageio import AICSImage
import napari
import numpy as np
import scipy.ndimage as ndi
import skimage.filters as ski_fil
import skimage.morphology as ski_mor
import skimage.exposure as ski_exp
import skimage.measure as ski_mea
import skimage.feature as ski_fea
import seaborn as sbn
import matplotlib.pyplot as plt
import sklearn.neighbors as skl_nei
import scipy.optimize as sp_opt
import os.path as osp
import pandas as pd


def contrast_stretch_by_ch(data):
    print("Contrast stretching by channel...")
    for ch in range(data.shape[0]):
        print("- Contrast stretching channel {}...".format(ch), end="")
        data[ch, ...] = ski_exp.rescale_intensity(data[ch, ...])
        print("done!")
    return data


def uint16_to_uint8(data):
    print("Converting from uint16 to uint8...", end="")
    converted = data // 256
    converted = converted.astype(np.uint8)
    print("done!")
    return converted


# Specify channels
NUCLEI_CH = 0
FISH_647_CH = 1
FISH_568_CH = 2

# Ask user to choose a file
filename = filedialog.askopenfilenames()[0]
print("--- Starting new analysis ---")
print("File: {}".format(filename))

# Load image and extract data
print("Loading...", end="")
original = AICSImage(filename)
data = original.get_image_data("CZYX")
print("done!")

pixel_sizes = original.physical_pixel_sizes
spacing = (pixel_sizes.Z, pixel_sizes.Y, pixel_sizes.X)
spacing_ratio = int(np.ceil(spacing[0] / spacing[1]))
contrast = [np.min(data), np.max(data)]
channels = original.channel_names
print("Original data info:")
print("- Image shape (CH, Z, Y, X): {}".format(data.shape))
print("- Pixel sizes (Z, Y, X): {}".format(spacing))
print("- Data type: {}".format(original.dtype))
print("- Data range: {}".format(contrast))
print("- Channels: {}".format(channels))

# Shrink data - for quicker development
dds = [d//4 for d in data.shape]
dde = [d - d//4 for d in data.shape]
data = data[:, dds[1]:dde[1], dds[2]:dde[2], dds[3]:dde[3]]

# Stretch each channel
data = contrast_stretch_by_ch(data)

# If needed, convert to uint8
if data.dtype != 'uint8':
    data = uint16_to_uint8(data)

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

# TODO: Should the intensities of the different slices be equalized due to the fact that at higher
#       Zs, we are deeper in the tissue?
#       If so, should we use regions without signal and just noise?
#       Should we also equalize across channels?

# Denoise
footprint_dim = [(3, 3, 3),
                 (3, 3, 3),
                 (3, 3, 3)]
footprint_dim[NUCLEI_CH] = (3, 3 * spacing_ratio, 3 * spacing_ratio)
denoised = np.empty_like(data)
print("Denoising...")
for ch in range(len(channels)):
    print("- Denoising channel {}...".format(ch), end="")
    file = osp.splitext(filename)
    file = file[0] + "-{}-den-{}-{}.npy".format(data.shape, ch, footprint_dim[ch])
    if osp.exists(file):
        den_ch = np.load(file)
    else:
        den_ch = ndi.median_filter(
            data[ch, ...],
            footprint=np.ones(footprint_dim[ch])
        )
        np.save(file, den_ch)
    denoised[ch, ...] = den_ch
    print("done!")
denoised = contrast_stretch_by_ch(denoised)
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

## Identify and segment the individual nuclei

# Threshold to identify the nuclei
print("Thresholding nuclei...", end="")
nuclei_mask = denoised[NUCLEI_CH] > ski_fil.threshold_otsu(denoised[NUCLEI_CH])
viewer.add_image(
    nuclei_mask,
    opacity=0.5,
    scale=spacing,
    colormap="blue",
    blending="additive",
    interpolation="nearest",
    visible=False
)
print("done!")

# TODO: ideas to try to detect center of nuclei:
#       - Blurr segmented, find max, watershed with Vornoi boundaries:
#         https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/Segmentation_3D.ipynb
#         https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/voronoi_otsu_labeling.ipynb)
#       - Gaussian blurr, then 3D Frangi to detect blobness then max of Frangi to detect nuclei
#         centers and then use those for watershed.
#       - Autocorrelation: cells are about the same size and autocorrelation should peak when
#         they overlap. Or use one cell as template and template matching. If we can find the
#         centers, we can then use watershed to identify individual cells.
#       - Consider a z-by-z segmentation with prior given from neighboring slices.

# Evaluate connected components
print("Evaluating nuclei connected components...")
nuclei_comp = ski_mea.label(nuclei_mask)
nuclei_props_df = pd.DataFrame(
    ski_mea.regionprops_table(
        nuclei_comp,
        properties=('label', 'area')
    )
)

# Evaluate and store image properties based on nuclei components
for ch in range(len(channels)):
    comp_props_df = pd.DataFrame(
        ski_mea.regionprops_table(
            nuclei_comp,
            intensity_image=data[ch, ...],
            properties=('label', 'intensity_mean')
        )
    )
    nuclei_props_df = nuclei_props_df.merge(comp_props_df, on='label')
    nuclei_props_df = nuclei_props_df.rename(columns={'intensity_mean': 'intensity_mean_ch{}'.format(channels[ch])})

# Evaluate nuclei's volume statistics and use to select individual nuclei
# NOTE: This only really works, when it works, with the large median filter in x and y.
nuclei_min_vol, nuclei_med_vol, nuclei_max_vol = nuclei_props_df['area'].quantile([0.25, 0.5, 0.75])
nuclei_mean = nuclei_props_df['area'].mean()
nuclei_std = nuclei_props_df['area'].std()
print(" - Volumes quantiles: {}, {}, {}".format(nuclei_min_vol, nuclei_med_vol, nuclei_max_vol))
print(" - Selected volume range: {} - {}".format(nuclei_min_vol, nuclei_max_vol))

# Show statistics
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

# Remove large and small components based on percentiles
print(" - Removing unwanted nuclei connected components...", end="")
nuclei_props_df.loc[:, 'keep'] = False
qry = 'area > ' + str(nuclei_min_vol) + ' & area < ' + str(nuclei_max_vol)
mask = nuclei_props_df.query(qry).index
nuclei_props_df.loc[mask, 'keep'] = True

# Generate the cleaned nuclei bool mask and the labeled components
nuclei_mask_cleaned = nuclei_mask.copy()
# nuclei_props_df.loc[:, 'id'] = 0
# comp_id = 1
# nuclei_labels = np.zeros_like(nuclei_mask, dtype='uint8')
for _, row in nuclei_props_df.iterrows():
    # print("{} -> {} ".format(row['label'], row['area']), end='')
    if not row['keep']:
        # print('x', end='')
        nuclei_mask_cleaned[nuclei_comp == row['label']] = False
    # else:
        # Assign an id to this nucleus and update the dataframe
        # nuclei_labels[nuclei_comp == row['label']] = comp_id
        # nuclei_props_df.loc[nuclei_props_df['label'] == row['label'], 'id'] = comp_id
        # comp_id += 1
    # print('')

viewer.add_image(
    nuclei_mask_cleaned,
    scale=spacing,
    opacity=0.5,
    colormap="blue",
    blending="additive",
    interpolation="nearest",
    visible=False
)
print("done!")

# Mask original nuclei channel
print("Masking nuclei channel...", end="")
nuclei_masked = data[NUCLEI_CH].copy()
nuclei_masked[nuclei_mask_cleaned==False] = 0
viewer.add_image(
    nuclei_masked,
    scale=spacing,
    opacity=1,
    colormap="green",
    blending="additive",
    interpolation="nearest",
    visible=False
)
print("done!")

# Mask original labels
print("Masking nuclei connected components...", end="")
# nuclei_labels = ski_mea.label(nuclei_mask_cleaned)
nuclei_labels = nuclei_comp.copy()
nuclei_labels[nuclei_mask_cleaned==False] = 0
viewer.add_labels(
    nuclei_labels,
    scale=spacing,
    blending="additive",
    visible=False
)
print("done!")

# Identify the FISH signals
print("Identifying 568nm channel puncta...", end="")
fish_568_blobs = ski_fea.blob_log(
    data[FISH_568_CH],
    min_sigma=1,
    max_sigma=10,
    num_sigma=10,
    threshold=0.05
)
fish_568_blobs_df = pd.DataFrame(fish_568_blobs, columns=['Z', 'Y', 'X', 'sigma'])
fish_568_blobs_df.loc[:, 'label'] = range(fish_568_blobs_df.shape[0])
print("{} detected...".format(len(fish_568_blobs)), end="")
viewer.add_points(
    fish_568_blobs[:, :3],
    size=5,
    symbol='disc',
    name='FISH_568',
    opacity=0.2,
    scale=spacing,
    edge_color='red',
    face_color='red',
    blending="additive",
    out_of_slice_display=True,
    visible=False
)
print("done!")

# Select FISH signatures within nuclei
print("Assigning 568nm puncta to nuclei...", end="")
fish_568_blobs_df.loc[:, ['keep', 'nucleus']] = False, None
nuclei_props_df.loc[:, channels[FISH_568_CH] + '_cnt'] = 0
nuclei_props_df[channels[FISH_568_CH] + '_ids'] = [[] for _ in range(nuclei_props_df.shape[0])]
for _, row in fish_568_blobs_df.iterrows():
    coo = row[['Z', 'Y', 'X']].astype('uint16')
    label = nuclei_labels[coo[0], coo[1], coo[2]]
    if label != 0:
        nuclei_props_df.loc[nuclei_props_df['label'] == label, channels[FISH_568_CH] + '_cnt'] += 1
        nuclei_props_df.loc[nuclei_props_df['label'] == label, channels[FISH_568_CH] + '_ids'].values[0].append(row['label'])
        fish_568_blobs_df.loc[fish_568_blobs_df['label'] == row['label'], ['keep', 'nucleus']] = True, label
print("done!")

# Create a point layer with only the puncta within nuclei











# print("- Masking...", end="")
# fish_568_mask = denoised[FISH_568_CH] > ski_fil.threshold_sauvola(fish_568_open, window_size=25)
# print("done!")
# viewer.add_image(
#     fish_568_erode,
#     opacity=0.5,
#     scale=spacing,
#     colormap="red",
#     blending="additive",
#     interpolation="nearest",
#     visible=False
# )
# viewer.add_image(
#     fish_568_open,
#     opacity=0.5,
#     scale=spacing,
#     colormap="red",
#     blending="additive",
#     interpolation="nearest",
#     visible=False
# )
# viewer.add_image(
#     fish_568_mask,
#     opacity=0.5,
#     scale=spacing,
#     colormap="red",
#     blending="additive",
#     interpolation="nearest",
#     visible=False
# )
# print("done!")










# # Mask all channels using nuclei
# print("Masking and contrast stretching...")
# masked = np.empty_like(data)
# for ch in range(len(channels)):
#     print("- Masking and stretching channel {}...".format(ch), end="")
#     msk_ch = denoised[ch, ...].copy()
#     msk_ch[nuclei_mask_cleaned == False] = 0
#     masked[ch, ...] = ski_exp.rescale_intensity(msk_ch)
#     print("done!")
#
# names = [c + '-den-msk' for c in channels]
# viewer.add_image(
#     masked,
#     channel_axis=0,
#     name=names,
#     scale=spacing,
#     colormap=["green", "magenta", "gray"],
#     blending="additive",
#     interpolation="nearest",
#     visible=False
# )
#
# print("FISH scoring...", end="")
# plt.figure()
# sbn.scatterplot(
#     nuclei_props_df[nuclei_props_df['keep']],
#     x='intensity_mean_ch0',
#     y='intensity_mean_ch1',
#     size='intensity_mean_ch2',
#     hue='label'
# )
# top = max(nuclei_props_df['intensity_mean_ch0'].max(), nuclei_props_df['intensity_mean_ch1'].max())
# plt.xlim((0, top))
# plt.ylim((0, top))
# plt.title(osp.split(filename)[1])
# plt.axis('square')
# plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
# plt.title('Average intensity-based FISH scoring')
# print("done!")


# Flush print buffer
print('', flush=True)

# Show
plt.show()
napari.run()

# TODO: Software to check:
#        - CellProfiler (https://cellprofiler.org/)
#        - CellSegm (https://scfbm.biomedcentral.com/articles/10.1186/1751-0473-8-16)
#        - SMMF algorithm (https://ieeexplore.ieee.org/document/4671118)
#        - Imaris methods