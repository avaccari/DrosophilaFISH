from tkinter import filedialog
from aicsimageio import AICSImage
import napari
import numpy as np
import scipy.ndimage as sci_ndi
import scipy.spatial as sci_spa
import skimage.filters as ski_fil
import skimage.exposure as ski_exp
import skimage.measure as ski_mea
import skimage.feature as ski_fea
import skimage.morphology as ski_mor
import seaborn as sbn
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd


def contrast_stretch_by_ch(data):
    print("Contrast stretching by channel...")
    for ch in range(data.shape[0]):
        print("- Contrast stretching channel {}...".format(ch), end="", flush=True)
        data[ch, ...] = ski_exp.rescale_intensity(data[ch, ...])
        print("done!")
    return data


def uint16_to_uint8(data):
    print("Converting from uint16 to uint8...", end="", flush=True)
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

# Load image and extract data
print("Loading file {}...".format(filename), end="", flush=True)
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
print("  - Spacing ratio (Z / X, Y): {}".format(spacing_ratio))
print("- Data type: {}".format(original.dtype))
print("- Data range: {}".format(contrast))
print("- Channels: {}".format(channels))

# --- Development only: shrink data ---
dds = [np.floor(d//5).astype('uint16') for d in data.shape]
dde = [np.ceil(d - d//5).astype('uint16') for d in data.shape]
data = data[:, dds[1]:dde[1], dds[2]:dde[2], dds[3]:dde[3]]
# -------------------------------------

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

# Denoise
footprint_dim = [(3, 3, 3),
                 (3, 3, 3),
                 (3, 3, 3)]
footprint_dim[NUCLEI_CH] = (3, 3 * spacing_ratio, 3 * spacing_ratio)
denoised = np.empty_like(data)
print("Denoising...")
for ch in range(len(channels)):
    print("- Denoising channel {}...".format(ch), end="", flush=True)
    file = osp.splitext(filename)
    file = file[0] + \
        "-{}-den-{}-{}.npy".format(data.shape, ch, footprint_dim[ch])
    if osp.exists(file):
        den_ch = np.load(file)
    else:
        den_ch = sci_ndi.median_filter(
            data[ch, ...],
            mode='constant',
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

# --- Identify and segment the individual nuclei ---
# Threshold to identify the nuclei
print("Thresholding nuclei...", end="", flush=True)
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

# Bounding surfaces for thresholded nuclei
print("Evaluating thresholded nuclei's boundaries...", end="", flush=True)
nuclei_mask_boundaries = ski_mor.dilation(
    nuclei_mask, footprint=ski_mor.ball(1)) ^ nuclei_mask
viewer.add_image(
    nuclei_mask_boundaries,
    opacity=0.5,
    scale=spacing,
    colormap="red",
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
#       - Idea for automated classification: after detecting a few, use a 3x3 block with some classic classification
#         method: KNN, SVM, etc. to classify the rest of the image. Use both the original image and the median-filtered
#         one to generate the features 3x3 from both. Also other features could include gradient and laplacian of the
#         image.
#       - You really have to try Random Forest of Extra Tree. Think about a reasonable set of features to use that might
#         help identify inside and boundaries. Also check the libraries that are available in Python to extract features
#         "Deep learning meets radiomics for end-to-end brain tumor mri analysis" (W. Ponikiewski)
#       - Check ICIP2022 paper #1147 (and the following on in the panel - Greek guy)

# Evaluate connected components
print("Evaluating nuclei connected components...")
nuclei_comp = ski_mea.label(nuclei_mask)
nuclei_props_df = pd.DataFrame(
    ski_mea.regionprops_table(
        nuclei_comp,
        properties=('label',
                    'area',
                    'axis_major_length',
                    'axis_minor_length',
                    'bbox',
                    'equivalent_diameter_area',
                    'slice',
                    'solidity')
    )
)

#
# # Evaluate and store image properties based on nuclei components
# for ch in range(len(channels)):
#     comp_props_df = pd.DataFrame(
#         ski_mea.regionprops_table(
#             nuclei_comp,
#             intensity_image=data[ch, ...],
#             properties=('label', 'intensity_mean')
#         )
#     )
#     nuclei_props_df = nuclei_props_df.merge(comp_props_df, on='label')
#     nuclei_props_df = nuclei_props_df.rename(columns={'intensity_mean': 'intensity_mean_ch{}'.format(channels[ch])})

# Evaluate nuclei's volume statistics and use to select individual nuclei
# NOTE: This only really works, when it works, with the large median filter in x and y.
nuclei_min_vol, nuclei_med_vol, nuclei_max_vol = nuclei_props_df['area'].quantile([
                                                                                  0.25, 0.5, 0.75])
nuclei_mean = nuclei_props_df['area'].mean()
nuclei_std = nuclei_props_df['area'].std()
print(" - Volumes quantiles: {}, {}, {}".format(
    nuclei_min_vol,
    nuclei_med_vol,
    nuclei_max_vol
)
)
print(" - Selected volume range: {} - {}".format(
    nuclei_min_vol,
    nuclei_max_vol)
)

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
print(" - Removing unwanted nuclei connected components...", end="", flush=True)
nuclei_props_df.loc[:, 'keep'] = False
qry = 'area > ' + str(nuclei_min_vol) + ' & area < ' + str(nuclei_max_vol)
mask = nuclei_props_df.query(qry).index
nuclei_props_df.loc[mask, 'keep'] = True

# Generate the cleaned nuclei bool mask and the labeled components
nuclei_mask_cleaned = nuclei_mask.copy()
for _, row in nuclei_props_df[nuclei_props_df['keep'] == False].iterrows():
    nuclei_mask_cleaned[nuclei_comp == row['label']] = False

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

print("Nuclei's dataframe:\n", nuclei_props_df)

# Bounding surfaces for uniquely detected nuclei
print("Evaluating uniquely detected nuclei's boundaries...", end="", flush=True)
nuclei_mask_cleaned_boundaries = ski_mor.dilation(
    nuclei_mask_cleaned, footprint=ski_mor.ball(1)) ^ nuclei_mask_cleaned
viewer.add_image(
    nuclei_mask_cleaned_boundaries,
    opacity=0.5,
    scale=spacing,
    colormap="red",
    blending="additive",
    interpolation="nearest",
    visible=False
)
print("done!")

# Mask original nuclei channel
print("Masking nuclei channel...", end="", flush=True)
nuclei_masked = data[NUCLEI_CH].copy()
nuclei_masked[nuclei_mask_cleaned == False] = 0
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
print("Masking nuclei connected components...", end="", flush=True)
# nuclei_labels = ski_mea.label(nuclei_mask_cleaned)
nuclei_labels = nuclei_comp.copy()
nuclei_labels[nuclei_mask_cleaned == False] = 0
viewer.add_labels(
    nuclei_labels,
    scale=spacing,
    blending="additive",
    visible=False
)
print("done!")


################################################################################
# Hack to be removed: Dilate preserved nuclei labels to identify nearby puncta
# nuclei_labels = ski_mor.dilation(nuclei_labels, footprint=ski_mor.ball(3))
################################################################################


# Use LOG to detect location of nuclei
print("Detecting nuclei centers...", end="", flush=True)
nuclei_ctrs = ski_fea.blob_log(
    data[NUCLEI_CH],
    min_sigma=20 * np.array((1 / spacing_ratio, 1, 1)),
    max_sigma=20 * np.array((1 / spacing_ratio, 1, 1)),
    num_sigma=1,
    threshold=0.05,
    exclude_border=True
)

print(nuclei_ctrs)

viewer.add_points(
    nuclei_ctrs[:, :3],
    name='nuclei_centers',
    size=5,
    symbol='disc',
    opacity=1,
    scale=spacing,
    edge_color='green',
    face_color='green',
    blending="additive",
    out_of_slice_display=True,
    visible=False
)
print("done!")


# Find the equivalent to Voronoi regions based on detected nuclei's centers
# Coordinates are normalized to the physical size before evaluating the regions
print("Identifying Voronoi regions...", end="", flush=True)
nuclei_ctrs_tree = sci_spa.KDTree(nuclei_ctrs[:, :3] * (spacing_ratio, 1, 1))
nuclei_mask_idx = nuclei_mask.nonzero()
nuclei_mask_idx_array = np.vstack(nuclei_mask_idx).T
closest_nucleus = nuclei_ctrs_tree.query(
    nuclei_mask_idx_array * (spacing_ratio, 1, 1))
nuclei_labels_voronoi = np.zeros_like(data[NUCLEI_CH])
nuclei_labels_voronoi[nuclei_mask_idx[0], nuclei_mask_idx[1],
                      nuclei_mask_idx[2]] = closest_nucleus[1] + 1
viewer.add_labels(
    nuclei_labels_voronoi,
    scale=spacing,
    blending="additive",
    visible=False
)
print("done!")


plt.show()
napari.run()


# --- Identify FISH puncta in the other channels ---
# -- Identify the 568 FISH signals --
# TODO: save the detected puncta to file, maybe the dataframes
print("Identifying 568nm channel puncta...", end="", flush=True)
fish_568_puncta = ski_fea.blob_log(
    data[FISH_568_CH],
    min_sigma=4 * np.array((1 / spacing_ratio, 1, 1)),
    max_sigma=7 * np.array((1 / spacing_ratio, 1, 1)),
    num_sigma=10,
    threshold=0.01,
    exclude_border=True
)
fish_568_puncta_df = pd.DataFrame(
    fish_568_puncta, columns=['Z', 'Y', 'X', 'sigma'])
fish_568_puncta_df.loc[:, 'label'] = range(fish_568_puncta_df.shape[0])
print("{} detected".format(len(fish_568_puncta)))
print("sigmas:\n", fish_568_puncta_df['sigma'].describe())
viewer.add_points(
    fish_568_puncta[:, :3],
    name='fish_568_puncta',
    size=5,
    symbol='disc',
    opacity=0.2,
    scale=spacing,
    edge_color='red',
    face_color='red',
    blending="additive",
    out_of_slice_display=True,
    visible=False
)
print("done!")


# Select 568 FISH signatures within nuclei
print("Assigning 568nm puncta to nuclei...", end="", flush=True)
fish_568_puncta_df.loc[:, ['keep', 'nucleus']] = False, None
nuclei_props_df.loc[:, channels[FISH_568_CH] + '_cnt'] = 0
nuclei_props_df[channels[FISH_568_CH] + '_ids'] = [[]
                                                   for _ in range(nuclei_props_df.shape[0])]
for _, row in fish_568_puncta_df.iterrows():
    coo = row[['Z', 'Y', 'X']].astype('uint16')
    label = nuclei_labels[coo[0], coo[1], coo[2]]
    if label != 0:
        nuclei_props_df.loc[nuclei_props_df['label'] ==
                            label, channels[FISH_568_CH] + '_cnt'] += 1
        nuclei_props_df.loc[nuclei_props_df['label'] == label,
                            channels[FISH_568_CH] + '_ids'].values[0].append(row['label'])
        fish_568_puncta_df.loc[fish_568_puncta_df['label']
                               == row['label'], ['keep', 'nucleus']] = True, label
print("done!")

# Create a point layer with only the 568 puncta within nuclei
fish_568_puncta_masked = fish_568_puncta_df.loc[fish_568_puncta_df['keep'], [
    'Z', 'Y', 'X']].to_numpy()
viewer.add_points(
    fish_568_puncta_masked,
    size=5,
    symbol='disc',
    opacity=0.2,
    scale=spacing,
    edge_color='red',
    face_color='red',
    blending="additive",
    out_of_slice_display=True,
    visible=False
)

# -- Identify the 647 FISH signals --
print("Identifying 647nm channel puncta...", end="", flush=True)
fish_647_puncta = ski_fea.blob_log(
    data[FISH_647_CH],
    min_sigma=4 * np.array((1 / spacing_ratio, 1, 1)),
    max_sigma=7 * np.array((1 / spacing_ratio, 1, 1)),
    num_sigma=10,
    threshold=0.01,
    exclude_border=True
)
fish_647_puncta_df = pd.DataFrame(
    fish_647_puncta, columns=['Z', 'Y', 'X', 'sigma'])
fish_647_puncta_df.loc[:, 'label'] = range(fish_647_puncta_df.shape[0])
print("{} detected".format(len(fish_647_puncta)))
print("sigmas:\n", fish_647_puncta_df['sigma'].describe())
viewer.add_points(
    fish_647_puncta[:, :3],
    name='fish_647_puncta',
    size=5,
    symbol='disc',
    opacity=0.2,
    scale=spacing,
    edge_color='blue',
    face_color='blue',
    blending="additive",
    out_of_slice_display=True,
    visible=False
)
print("done!")

# Select 647 FISH signatures within nuclei
print("Assigning 647nm puncta to nuclei...", end="", flush=True)
fish_647_puncta_df.loc[:, ['keep', 'nucleus']] = False, None
nuclei_props_df.loc[:, channels[FISH_647_CH] + '_cnt'] = 0
nuclei_props_df[channels[FISH_647_CH] + '_ids'] = [[]
                                                   for _ in range(nuclei_props_df.shape[0])]
for _, row in fish_647_puncta_df.iterrows():
    coo = row[['Z', 'Y', 'X']].astype('uint16')
    label = nuclei_labels[coo[0], coo[1], coo[2]]
    if label != 0:
        nuclei_props_df.loc[nuclei_props_df['label'] ==
                            label, channels[FISH_647_CH] + '_cnt'] += 1
        nuclei_props_df.loc[nuclei_props_df['label'] == label,
                            channels[FISH_647_CH] + '_ids'].values[0].append(row['label'])
        fish_647_puncta_df.loc[fish_647_puncta_df['label']
                               == row['label'], ['keep', 'nucleus']] = True, label
print("done!")

# Create a point layer with only the 647 puncta within nuclei
fish_647_puncta_masked = fish_647_puncta_df.loc[fish_647_puncta_df['keep'], [
    'Z', 'Y', 'X']].to_numpy()
viewer.add_points(
    fish_647_puncta_masked,
    size=5,
    symbol='disc',
    opacity=0.2,
    scale=spacing,
    edge_color='blue',
    face_color='blue',
    blending="additive",
    out_of_slice_display=True,
    visible=False
)

plt.figure()
sbn.scatterplot(
    nuclei_props_df[nuclei_props_df['keep']],
    x=channels[FISH_568_CH] + '_cnt',
    y=channels[FISH_647_CH] + '_cnt',
)
top = max(nuclei_props_df[channels[FISH_568_CH] + '_cnt'].max(),
          nuclei_props_df[channels[FISH_647_CH] + '_cnt'].max())
plt.xlim((0, top))
plt.ylim((0, top))
plt.title(osp.split(filename)[1])
plt.axis('square')
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

# Save the dataframes
print("Saving the nuclei's and puncta property dataframe...", end="", flush=True)
file = osp.splitext(filename)
nuclei_props_df.to_json(file[0] + "-{}-nuclei_df.json".format(data.shape))
fish_568_puncta_df.to_json(file[0] + "-{}-568_df.json".format(data.shape))
fish_647_puncta_df.to_json(file[0] + "-{}-647_df.json".format(data.shape))
print("done!")


plt.show()
napari.run()

# TODO: Software to check:
#        - CellProfiler (https://cellprofiler.org/)
#        - CellSegm (https://scfbm.biomedcentral.com/articles/10.1186/1751-0473-8-16)
#        - SMMF algorithm (https://ieeexplore.ieee.org/document/4671118)
#        - Imaris methods
#        - 3D UNET, nnUNET
#        - pyradiomics (https://pyradiomics.readthedocs.io/en/latest/index.html) for feature extraction
