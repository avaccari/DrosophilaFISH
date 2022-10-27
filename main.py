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


class Channel:
    # A counter for the channels
    _channel_number = 0

    # A static method to contrast stretch
    def contrast_stretch(data):
        return ski_exp.rescale_intensity(data)

    def __init__(self, data, voxel_size=(1, 1, 1), name=None, contrast_stretch=True, convert_to_uint8=True):
        self.data = data
        self.voxel_size = voxel_size
        self._z_ratio = int(np.ceil(voxel_size[0] / voxel_size[1]))
        if name is None:
            self.name = 'Channel' + str(Channel.channel_number)
        else:
            self.name = name
        self._channel_number = Channel._channel_number
        Channel._channel_number += 1

        if contrast_stretch:
            print("Contrast stretching channel {}...".format(
                self.name), end="", flush=True)
            self.data = Channel.contrast_stretch(self.data)
            print("done!")

        if convert_to_uint8:
            if self.data.dtype == 'uint8':
                print("\nData already in `uint8` format. Conversion skipped.")
            else:
                if self.data.dtype != 'uint16':
                    print(
                        "\nWARNING: Conversion from types other than 'uint16' not implemented!")
                else:
                    print("Converting from uint16 to uint8 channel {}...".format(
                        self.name), end="", flush=True)
                    self.data = (self.data // 256).astype('uint8')
                    print("done!")

        self.original = self.data.copy()

    def median_filter(self, footprint=None, file_root=None, contrast_stretch=True):
        # TODO: Should the intensities of the different slices be equalized due
        # to the fact that at higher Zs, we are deeper in the tissue? If so,
        # should we use regions without signal and just noise? Should we also
        # equalize across channels?
        if footprint is None:
            footprint_dim = (3, 3 * self._z_ratio, 3 * self._z_ratio)
            footprint = np.ones(footprint_dim)
        else:
            footprint_dim = footprint.shape

        print("Applying median filter to channel {}...".format(
            self.name), end="", flush=True)

        if file_root is not None:
            file = osp.splitext(file_root)
            file = file[0] + "-{}-{}-den-{}.npy".format(
                self.name,
                self.data.shape,
                footprint_dim
            )
            try:
                filtered = np.load(file)
            except OSError:
                filtered = sci_ndi.median_filter(
                    self.data,
                    mode='constant',
                    footprint=footprint
                )
                try:
                    np.save(file, filtered)
                except:
                    print("\n\nWARNING: The file {} couldn't be saved!\n".format(file))
        else:
            filtered = sci_ndi.median_filter(
                self.data,
                mode='constant',
                footprint=footprint
            )

        print("done!")

        if contrast_stretch:
            print("Contrast stretching...", end="", flush=True)
            filtered = Channel.contrast_stretch(filtered)
            print("done!")

        return filtered


class Image:
    def __init__(self, filename=None, channel_names=None, contrast_stretch=True, convert_to_uint8=True):
        self.channels = []

        if filename is None:
            filename = filedialog.askopenfilenames()[0]

        print("\n\n--- Starting new analysis ---")

        print("Loading file {}...".format(filename), end="", flush=True)
        original = AICSImage(filename)
        data = original.get_image_data("CZYX")
        print("done!")

        self.channels_no = data.shape[0]
        self.original_dtype = data.dtype

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
        print("- Original channels names: {}".format(channels))

        if len(channels) != len(channel_names):
            print("\nWARNING: the provided channels name list doesn't match the lenght of the channels in the image. Using original names.")
        else:
            channels = channel_names
            print("- assigned channels names: {}".format(channels))

        print("Parsing channels...")
        for ch in range(self.channels_no):
            self.channels.append(
                Channel(
                    data[ch, ...],
                    voxel_size=spacing,
                    name=channels[ch],
                    contrast_stretch=contrast_stretch,
                    convert_to_uint8=convert_to_uint8
                )
            )
        print("done!")


file = '/Volumes/GoogleDrive-104915504535925824452/Shared drives/MarkD/FISH analysis/FISH dataset (2x exp, 16 bit) Oct 2022/48h_GFP_dpr13-568_dpr17-647v_1(2x_16bit)_6.czi'
channel_names = ('Nuclei', 'Fish_647', 'Fish_568')
image = Image(file, channel_names=channel_names)

# Specify channels
NUCLEI_CH = 0
FISH_647_CH = 1
FISH_568_CH = 2

# --- Development only: shrink data ---
# dds = [np.floor(d//5).astype('uint16') for d in data.shape]
# dde = [np.ceil(d - d//5).astype('uint16') for d in data.shape]
# data = data[:, dds[1]:dde[1], dds[2]:dde[2], dds[3]:dde[3]]
# -------------------------------------


# Show original data
print("Adding original images to viewer...", end="", flush=True)
viewer = napari.Viewer(
    title=osp.split(file)[1],
    ndisplay=3
)
for ch in range(image.channels_no):
    viewer.add_image(
        image.channels[ch].data,
        name=image.channels[ch].name,
        colormap="green",
        blending="additive",
        scale=image.channels[ch].voxel_size,
        depiction="volume",
        interpolation="nearest",
        visible=False
    )
print("done!")

# Denoise and stretch the nuclei channel
filtered = image.channels[NUCLEI_CH].median_filter(file_root=file)
viewer.add_image(
    filtered,
    name=image.channels[NUCLEI_CH].name + "-den",
    colormap="magenta",
    blending="additive",
    scale=image.channels[NUCLEI_CH].voxel_size,
    interpolation="nearest",
    visible=False
)

napari.run()


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
