from tkinter import filedialog
from aicsimageio import AICSImage
import napari
import numpy as np
import scipy.ndimage as sci_ndi
import scipy.spatial as sci_spa
import scipy.stats as sci_sta
import skimage.filters as ski_fil
import skimage.filters.rank as ski_fil_ran
import skimage.exposure as ski_exp
import skimage.measure as ski_mea
import skimage.feature as ski_fea
import skimage.morphology as ski_mor
import skimage.segmentation as ski_seg
import skimage.io as ski_io
import seaborn as sbn
import matplotlib.pyplot as plt
import os
import os.path as osp
import pandas as pd
import json


def contrast_stretch(data):
    print("Contrast stretching...", end="", flush=True)
    data = ski_exp.rescale_intensity(data)
    print("done!")
    return data


def uint16_to_uint8(data):
    print("Converting from uint16 to uint8...", end="", flush=True)
    converted = data // 256
    converted = converted.astype(np.uint8)
    print("done!")
    return converted


def build_path(filename_root, suffix=None):
    dir, file = osp.split(filename_root)
    file_name, _ = osp.splitext(file)
    root_dir = osp.join(dir, file_name)
    if suffix is None:
        return root_dir
    else:
        file_out = osp.join(root_dir, file_name + suffix)
        return file_out


def filter(
    data,
    type="median",
    footprint=None,
    sigma=(2, 10, 10),
    mode="nearest",
    cval=0,
    filename_root=None,
    ch_id=None,
    stretch=False,
):
    filtered = data.copy()

    if type not in ["median", "gaussian", "closing", "maximum"]:
        print("WARNING: the specified mode is not available.")

    if footprint is None:
        footprint = ski_mor.ball(1)
        footprint_dim = footprint.shape
    else:
        footprint_dim = footprint.shape

    print(f"Applying {type} filter...", end="", flush=True)

    if filename_root is None:
        if type == "median":
            filtered = sci_ndi.median_filter(
                data, mode=mode, cval=cval, footprint=footprint
            )
        elif type == "gaussian":
            filtered = sci_ndi.gaussian_filter(data, mode=mode, cval=cval, sigma=sigma)
        elif type == "closing":
            filtered = ski_mor.closing(data, footprint=footprint)
        elif type == "maximum":
            filtered = ski_fil_ran.maximum(data, footprint=footprint)
        else:
            pass
    else:
        if ch_id is None:
            print(
                "WARNING: a ch_id should be provided to identify the channel. The data was not filtered."
            )
        else:
            if type == "median":
                file = build_path(
                    filename_root, f"-{ch_id}-den-med-{footprint_dim}.npy"
                )
            elif type == "gaussian":
                file = build_path(
                    filename_root,
                    f"-{ch_id}-den-gaus-{tuple(np.round(sigma, decimals=2))}.npy",
                )
            elif type == "closing":
                file = build_path(
                    filename_root,
                    f"-{ch_id}-clos-{footprint_dim}.npy",
                )
            elif type == "maximum":
                file = build_path(
                    filename_root,
                    f"-{ch_id}-max-{footprint_dim}.npy",
                )
            else:
                pass
            try:
                filtered = np.load(file)
            except FileNotFoundError:
                if type == "median":
                    filtered = sci_ndi.median_filter(
                        data, mode=mode, cval=cval, footprint=footprint
                    )
                elif type == "gaussian":
                    filtered = sci_ndi.gaussian_filter(
                        data, mode=mode, cval=cval, sigma=sigma
                    )
                elif type == "closing":
                    filtered = ski_mor.closing(data, footprint=footprint)
                elif type == "maximum":
                    filtered = ski_fil_ran.maximum(data, footprint=footprint)
                else:
                    pass
                try:
                    root_dir = build_path(filename_root)
                    if not osp.isdir(root_dir):
                        os.makedirs(root_dir)
                    np.save(file, filtered)
                except:
                    print("WARNING: error saving the file.")
    print("done!")

    if stretch:
        filtered = contrast_stretch(filtered)

    return filtered


def remove_floor(data, sigma=(20, 100, 100), filename_root=None, ch_id=None):
    noise_floor = filter(
        data, type="gaussian", sigma=sigma, filename_root=filename_root, ch_id=ch_id
    )
    defloored = data.astype("int16") - noise_floor.astype("int16")
    return np.maximum(defloored, 0).astype("uint8")


def detect_blobs(
    data,
    min_sigma=1,
    max_sigma=50,
    num_sigma=10,
    z_y_x_ratio=None,
    threshold=0.01,
    overlap=0.75,
    filename_root=None,
    ch_id=None,
):
    print("Detecting blobs' centers...", end="", flush=True)
    if z_y_x_ratio is not None:
        min_sigma = min_sigma * np.array(z_y_x_ratio)
        max_sigma = max_sigma * np.array(z_y_x_ratio)
    blobs_ctrs = np.zeros((1, 3))
    if filename_root is None:
        blobs_ctrs = ski_fea.blob_log(
            data,
            min_sigma=min_sigma,
            max_sigma=max_sigma,
            num_sigma=num_sigma,
            threshold=threshold,
            overlap=overlap,
            exclude_border=True,
        )
    else:
        if ch_id is None:
            print(
                "WARNING: a ch_id should be provided to identify the channel. Blobs were not detected."
            )
        else:
            file = build_path(filename_root, f"-{ch_id}-blb.npy")
            try:
                blobs_ctrs = np.load(file)
            except FileNotFoundError:
                blobs_ctrs = ski_fea.blob_log(
                    data,
                    min_sigma=min_sigma,
                    max_sigma=max_sigma,
                    num_sigma=num_sigma,
                    threshold=threshold,
                    exclude_border=True,
                )
                try:
                    root_dir = build_path(filename_root)
                    if not osp.isdir(root_dir):
                        os.makedirs(root_dir)
                    np.save(file, blobs_ctrs)
                except:
                    print("WARNING: error saving the file.")
    print("done!")

    return blobs_ctrs


def load_metadata(filename_root, sec_data=None):
    file = build_path(filename_root, "-meta.json")
    try:
        with open(file) as meta_file:
            meta_data = json.load(meta_file)
            if sec_data is None:
                return meta_data
            else:
                return meta_data[sec_data]
    except OSError:
        print("WARNING: Metadata file not found!")
        return None
    except json.decoder.JSONDecodeError:
        print("WARNING: Metadata file is not in the correct format!")
        return None
    except KeyError:
        print(f"WARNING: Section {sec_data} not found in metadata file!")
        return {}


def save_metadata(filename_root, sec_name=None, sec_data=None):
    file = build_path(filename_root, "-meta.json")
    try:
        root_dir = build_path(filename_root)
        if not osp.isdir(root_dir):
            os.makedirs(root_dir)
        with open(file, "w") as meta_file:
            if meta_data is None:
                meta_data = {sec_name: sec_data}
            else:
                meta_data[sec_name] = sec_data
            json.dump(meta_data, meta_file)
    except OSError:
        print("WARNING: Issues opening the metadata file!")


def evaluate_voronoi(mask, markers, spacing=(1, 1, 1), filename_root=None, ch_id=None):
    # Find the equivalent to Voronoi regions in the provided mask based on the
    # provided markers
    # Coordinates are normalized to the physical size using 'spacing'
    print("Identifying Voronoi regions within mask...", end="", flush=True)
    if filename_root is None:
        markers_tree = sci_spa.KDTree(markers * spacing)
        mask_idx = mask.nonzero()
        mask_idx_array = np.vstack(mask_idx).T
        closest_marker = markers_tree.query(mask_idx_array * spacing)
        labels = np.zeros_like(mask, dtype="uint16")
        # The +1 is to start the labels at 1 instead of 0
        labels[mask_idx[0], mask_idx[1], mask_idx[2]] = closest_marker[1] + 1
    else:
        if ch_id is None:
            print(
                "WARNING: a ch_id should be provided to identify the channel. Blobs were not detected."
            )
        else:
            file = build_path(filename_root, f"-{ch_id}-voronoi.npy")
            try:
                labels = np.load(file)
            except FileNotFoundError:
                markers_tree = sci_spa.KDTree(markers * spacing)
                mask_idx = mask.nonzero()
                mask_idx_array = np.vstack(mask_idx).T
                closest_marker = markers_tree.query(mask_idx_array * spacing)
                labels = np.zeros_like(mask, dtype="uint16")
                # The +1 is to start the labels at 1 instead of 0
                labels[mask_idx[0], mask_idx[1], mask_idx[2]] = closest_marker[1] + 1
                try:
                    root_dir = build_path(filename_root)
                    if not osp.isdir(root_dir):
                        os.makedirs(root_dir)
                    np.save(file, labels)
                except:
                    print("WARNING: error saving the file.")

    print("done!")

    return labels


def analyze_image(filename=None, visualize=False):
    # Ask user to choose a file
    print("\n--- Starting new analysis ---")
    if filename is None:
        filename = filedialog.askopenfilename()

    # Load image and extract data
    print(f"Loading file {filename}...", end="", flush=True)
    original = AICSImage(filename)
    data = original.get_image_data("CZYX")
    print("done!")

    # Specify channels
    # TODO: since it can change from image to image, do it through metadata
    NUCLEI_CH = 0
    FISH_647_CH = 1
    FISH_568_CH = 2
    CYTO_CH = 3
    ch_dict = {
        NUCLEI_CH: "Nuclei",
        FISH_647_CH: "FISH_647",
        FISH_568_CH: "FISH_568",
        CYTO_CH: "Cytoplasm",
    }

    # Specify thresholds for FISH detection
    FISH_THRESHOLD_MIN = 5
    FISH_THRESHOLD_MAX = 251
    FISH_THRESHOLD_STEP = 5

    # Gather and report image information
    pixel_sizes = original.physical_pixel_sizes
    spacing = (pixel_sizes.Z, pixel_sizes.Y, pixel_sizes.X)
    spacing_ratio = int(np.ceil(spacing[0] / spacing[1]))
    contrast = [[np.min(data[ch]), np.max(data[ch])] for ch in range(data.shape[0])]
    channels = original.channel_names
    print("Original data info:")
    print(f"- Image shape (CH, Z, Y, X): {data.shape}")
    print(f"- Pixel sizes (Z, Y, X): {spacing}")
    print(f"  - Spacing ratio (Z / X, Y): {spacing_ratio}")
    print(f"- Data type: {original.dtype}")
    print(f"- Data range (per channel): {contrast}")
    print(f"- Channels: {channels} ({[v for v in ch_dict.values()]})")

    # --- Development only: shrink data ---
    # dds = [np.floor(d//5).astype('uint16') for d in data.shape]
    # dde = [np.ceil(d - d//5).astype('uint16') for d in data.shape]
    # data = data[:, dds[1]:dde[1], dds[2]:dde[2], dds[3]:dde[3]]
    # -------------------------------------

    # Contrast stretch
    for ch in range(data.shape[0]):
        print(f"Processing channel {ch_dict[ch]}:")
        data[ch] = contrast_stretch(data[ch])

    # If needed, convert to uint8
    if data.dtype != "uint8":
        data = uint16_to_uint8(data)

    # Remove floor from each channel and contrast stretch
    for ch in range(data.shape[0]):
        print(f"Removing floor from channel {ch_dict[ch]}:")
        data[ch] = remove_floor(
            data[ch],
            sigma=100 * np.array((1 / spacing_ratio, 1, 1)),
            filename_root=filename,
            ch_id=ch_dict[ch],
        )
        data[ch] = contrast_stretch(data[ch])

    if visualize:
        # Show original data
        viewer = napari.Viewer(title=osp.split(filename)[1], ndisplay=3)
        viewer.add_image(
            data,
            channel_axis=0,
            name=ch_dict.values(),
            colormap="green",
            blending="additive",
            scale=spacing,
            depiction="volume",
            interpolation="nearest",
            visible=False,
        )

    # # Apply a local histogram stretching to the nuclei channel
    # print("Local histogram stretching nuclei's channel...", end="", flush=True)
    # nuclei_hist = ski_exp.equalize_adapthist(
    #     nuclei_defloored,
    #     15 * np.array((1 / spacing_ratio, 1, 1))
    # )
    # nuclei_hist = np.round(255 * nuclei_hist).astype('uint8')
    # viewer.add_image(
    #     nuclei_hist,
    #     name=ch_dict[NUCLEI_CH] + '-hist',
    #     colormap="magenta",
    #     blending="additive",
    #     scale=spacing,
    #     interpolation="nearest",
    #     visible=False
    # )
    # print("done!")

    # Apply median filter to denoise the nuclei channel
    print("Denoising nuclei's channel:")
    nuclei_den = filter(
        data[NUCLEI_CH],
        footprint=ski_mor.ball(7)[3::4],
        filename_root=filename,
        ch_id=ch_dict[NUCLEI_CH],  # ! We removed the stretch
    )
    if visualize:
        viewer.add_image(
            nuclei_den,
            name=ch_dict[NUCLEI_CH] + "-den",
            colormap="magenta",
            blending="additive",
            scale=spacing,
            interpolation="nearest",
            visible=False,
        )

    # Semantic segmentation to identify all nuclei
    print("Thresholding nuclei...", end="", flush=True)
    nuclei_mask = nuclei_den > ski_fil.threshold_otsu(nuclei_den)
    # TODO: try the minimum error thresholding:
    #       http://www.cyto.purdue.edu/cdroms/micro2/content/education/wirth04.pdf
    if visualize:
        viewer.add_image(
            nuclei_mask,
            name=ch_dict[NUCLEI_CH] + "-msk",
            opacity=0.5,
            scale=spacing,
            colormap="blue",
            blending="additive",
            interpolation="nearest",
            visible=False,
        )
    print("done!")

    # # Apply closing to nuclei's mask
    # print("Closing nuclei's mask:")
    # nuclei_closed = filter(
    #     nuclei_mask,
    #     type="closing",
    #     footprint=ski_mor.ball(7)[3::4, ...],
    #     filename_root=filename,
    #     ch_id=ch_dict[NUCLEI_CH]
    # )
    # viewer.add_image(
    #     nuclei_closed,
    #     name=ch_dict[NUCLEI_CH] + '-closed',
    #     colormap="magenta",
    #     blending="additive",
    #     scale=spacing,
    #     interpolation="nearest",
    #     visible=False
    # )

    # TODO: following the threshold with the nearest neighbor creates some regions
    #       that have the same ID but are segmented. There might be multiple
    #       fractions of a nucleus that are identified while there is a single
    #       center that is detected. They are all the closest, but they are many.
    #       The result is an error while evaluating the properties.
    #       Should remove the smallest regions before the nearest neighbors.

    # TODO: evaluate the actual physical dimensions of the nuclei and use the
    #       voxel dimension info to decide the sigma of the radius. Maybe use a
    #       small range around the expected value.

    # TODO: might be good to use a range of sigmas but it requires more memory or
    #       some cropping before analysis. Could use the bbox of the thresholded
    #       nuclei, but it might miss some low intensity ones. Might be good anyway.

    # Detect the centers of the nuclei
    print("Detecting nuclei's centers:")
    nuclei_ctrs = detect_blobs(
        nuclei_den.astype("float"),
        min_sigma=25,
        max_sigma=25,
        num_sigma=1,
        z_y_x_ratio=(1 / spacing_ratio, 1, 1),
        threshold=5,
        filename_root=filename,
        ch_id=ch_dict[NUCLEI_CH],
    )

    # # Create dataframe with nuclei centers and assign IDs
    # nuclei_ctrs_df = pd.DataFrame(
    #     nuclei_ctrs,
    #     columns=['Z', 'Y', 'X', 'sigmaZ', 'sigmaY', 'sigmaX']
    #     )
    # nuclei_ctrs_df['ID'] = range(1, 1 + len(nuclei_ctrs))

    print(f"Detected {len(nuclei_ctrs)} centers:\n{nuclei_ctrs}")

    if visualize:
        viewer.add_points(
            nuclei_ctrs[:, :3],
            name=ch_dict[NUCLEI_CH] + "-ctrs",
            size=5,
            symbol="disc",
            opacity=1,
            scale=spacing,
            edge_color="green",
            face_color="green",
            blending="additive",
            out_of_slice_display=True,
            visible=False,
        )

    # Find the Voronoi regions in the nuclei's mask using detected nuclei's
    # centers as markers
    nuclei_labels_voronoi = evaluate_voronoi(
        nuclei_mask,
        nuclei_ctrs[:, :3],
        spacing=(spacing_ratio, 1, 1),
        filename_root=filename,
        ch_id=ch_dict[NUCLEI_CH],
    )

    if visualize:
        viewer.add_labels(
            nuclei_labels_voronoi,
            name=ch_dict[NUCLEI_CH] + "-vor-lbls",
            scale=spacing,
            blending="additive",
            visible=False,
        )

    # Use watershed to identify regions connected to the nuclei centers
    print(
        "Identifying regions within nuclei's mask with watershed...", end="", flush=True
    )

    # Create the markers
    nuclei_ctrs_img = np.zeros_like(data[NUCLEI_CH], dtype="uint16")
    nuclei_ctrs_img[
        nuclei_ctrs[:, 0].astype("uint16"),
        nuclei_ctrs[:, 1].astype("uint16"),
        nuclei_ctrs[:, 2].astype("uint16"),
    ] = range(1, 1 + len(nuclei_ctrs))

    # Remove markers that are not contained in the nuclei's mask
    nuclei_ctrs_img_clean = nuclei_ctrs_img & 65535 * nuclei_mask.astype("uint16")

    # Watershed nuclei's mask
    nuclei_labels_watershed = ski_seg.watershed(
        nuclei_mask, nuclei_ctrs_img_clean, mask=nuclei_mask, compactness=0
    ).astype("uint16")
    if visualize:
        viewer.add_labels(
            nuclei_labels_watershed,
            name=ch_dict[NUCLEI_CH] + "-ws-lbls",
            scale=spacing,
            blending="additive",
            visible=False,
        )
    print("done!")

    # Final labeled nuclei mask (no cytoplasm)
    print("Combining vornoi and watershed nuclei's labeling...", end="", flush=True)
    nuclei_labels = nuclei_labels_voronoi.copy()
    nuclei_labels[nuclei_labels_voronoi != nuclei_labels_watershed] = 0
    if visualize:
        viewer.add_labels(
            nuclei_labels,
            name=ch_dict[NUCLEI_CH] + "-lbls-no-cyto",
            scale=spacing,
            blending="additive",
            visible=False,
        )
    print("done!")

    # If cytoplasm channel exists, use it
    if data.shape[0] == 4:
        # Apply median filter to denoise the cytoplasm channel
        print("Denoising cytoplasm's channel:")
        cyto_den = filter(
            data[CYTO_CH],
            footprint=ski_mor.ball(7)[3::4],
            filename_root=filename,
            ch_id=ch_dict[CYTO_CH],
        )
        if visualize:
            viewer.add_image(
                cyto_den,
                name=ch_dict[CYTO_CH] + "-den",
                colormap="magenta",
                blending="additive",
                scale=spacing,
                interpolation="nearest",
                visible=False,
            )

        # Apply closing to denoised cytoplasm channel
        print("Closing cytoplasm's channel:")
        cyto_closed = filter(
            cyto_den,
            type="closing",
            footprint=ski_mor.ball(7)[3::4],
            filename_root=filename,
            ch_id=ch_dict[CYTO_CH],
        )
        if visualize:
            viewer.add_image(
                cyto_closed,
                name=ch_dict[CYTO_CH] + "-closed",
                colormap="magenta",
                blending="additive",
                scale=spacing,
                interpolation="nearest",
                visible=False,
            )

        # Semantic segmentation to identify all cytoplasm
        print("Thresholding cytoplasm...", end="", flush=True)
        cyto_mask = cyto_closed > 0
        if visualize:
            viewer.add_image(
                cyto_mask,
                name=ch_dict[CYTO_CH] + "-msk",
                opacity=0.5,
                scale=spacing,
                colormap="blue",
                blending="additive",
                interpolation="nearest",
                visible=False,
            )
        print("done!")

        # Invert the cytoplasm mask to identify potential nuclei
        print("Inverting cytoplasm mask...", end="", flush=True)
        cyto_mask_inv = ~cyto_mask
        if visualize:
            viewer.add_image(
                cyto_mask_inv,
                name=ch_dict[CYTO_CH] + "-msk-inv",
                opacity=0.5,
                scale=spacing,
                colormap="blue",
                blending="additive",
                interpolation="nearest",
                visible=False,
            )
        print("done!")

        # Find the Voronoi regions in the cytoplasm's mask using detected
        # nuclei's centers as markers
        cyto_inv_labels_voronoi = evaluate_voronoi(
            cyto_mask_inv,
            nuclei_ctrs[:, :3],
            spacing=(spacing_ratio, 1, 1),
            filename_root=filename,
            ch_id=ch_dict[CYTO_CH],
        )

        # Use watershed to identify regions connected to the nuclei centers
        print(
            "Identifying regions within inverse cytoplasm's mask with watershed...",
            end="",
            flush=True,
        )

        # Remove markers that are not contained in the inverse cytoplasm's mask
        nuclei_ctrs_img_clean = nuclei_ctrs_img & 65535 * cyto_mask_inv.astype("uint16")

        # Watershed nuclei's mask
        cyto_inv_labels_watershed = ski_seg.watershed(
            cyto_mask_inv, nuclei_ctrs_img_clean, mask=cyto_mask_inv, compactness=0
        ).astype("uint16")
        if visualize:
            viewer.add_labels(
                cyto_inv_labels_watershed,
                name=ch_dict[CYTO_CH] + "-inv-ws-lbls",
                scale=spacing,
                blending="additive",
                visible=False,
            )
        print("done!")

        # Final labeled nuclei mask (with cytoplasm)
        print(
            "Combining nuclei's labeling with inverse cytoplasm labeling...",
            end="",
            flush=True,
        )
        nuclei_labels[nuclei_labels != cyto_inv_labels_voronoi] = 0
        nuclei_labels[nuclei_labels != cyto_inv_labels_watershed] = 0
        if visualize:
            viewer.add_labels(
                nuclei_labels,
                name=ch_dict[NUCLEI_CH] + "-lbls-with-cyto",
                scale=spacing,
                blending="additive",
                visible=False,
            )
        print("done!")

        ############################################################################
        # Voronoi using the detected centers both for nuclei and inverted cyto.
        # Corresponding regions should have same id in both layers.
        # Then think about a way to combine them to optimize the segmentation.
        # Option1:
        # Optimize threshold level to equalize/maximize dice measure between nuclei
        # and cytoplasm masks using the masks obtained by using Voronoi on the
        # detected centroids. Should the smallest be adapted to match the largest or
        # should be a process where the threshold levels are adjusted until optimal
        # overal is found. At each iteration we might need to Voronoi again since we
        # are modifying the mask.
        ############################################################################

    ################################################################################
    # Hack to be improved: Dilate preserved nuclei labels to identify nearby puncta
    print("Dilate preserved nuclei to include part of the surrounding cytoplasm:")
    nuclei_labels = filter(
        nuclei_labels,
        type="maximum",
        footprint=ski_mor.ball(5)[2::3],
        filename_root=filename,
        ch_id=ch_dict[NUCLEI_CH],
    )
    if visualize:
        viewer.add_labels(
            nuclei_labels,
            name=ch_dict[NUCLEI_CH] + "-lbls-with-cyto-dilate",
            scale=spacing,
            blending="additive",
            visible=False,
        )
    ################################################################################

    # Evaluate potential nuclei properties
    # TODO: consider adding the detected centers and sigmas
    print("Evaluating potential nuclei properties from mask...", end="", flush=True)
    nuclei_props_df = pd.DataFrame(
        ski_mea.regionprops_table(
            nuclei_labels,
            properties=(
                "label",
                "area",
                "axis_major_length",
                "axis_minor_length",
                "equivalent_diameter_area",
                "slice",
                "solidity",
            ),
        )
    )
    nuclei_props_df.loc[:, "keep"] = True  # Keep all nuclei for now
    print("done!")

    print("Nuclei's properties:\n", nuclei_props_df)

    # Identify FISH puncta
    print("Identifying FISH puncta's centers within nuclei's bboxes...")
    # To minimize difference between channels, we are pre-building the images
    # with just the region that we will analyze and then contrast stretch
    # globally. This should normalize the puncta intensity among channels as
    # well as within the channels. Contrast stretching individual nuclei
    # area would be equivalent to use non uniform detection thresholds.
    print("Building images to analyze for FISH puncta...", end="", flush=True)
    fish_568_to_analyze = np.zeros_like(data[FISH_568_CH])
    fish_647_to_analyze = np.zeros_like(data[FISH_647_CH])
    min_intensity, max_intensity = np.inf, -np.inf
    for idx, row in nuclei_props_df[nuclei_props_df["keep"]].iterrows():
        sl = row["slice"]
        slice_647 = data[FISH_647_CH, sl[0], sl[1], sl[2]]
        slice_568 = data[FISH_568_CH, sl[0], sl[1], sl[2]]
        min_intensity = min(min_intensity, slice_647.min(), slice_568.min())
        max_intensity = max(max_intensity, slice_647.max(), slice_568.max())
        fish_647_to_analyze[sl[0], sl[1], sl[2]] = slice_647
        fish_568_to_analyze[sl[0], sl[1], sl[2]] = slice_568
    fish_568_to_analyze = ski_exp.rescale_intensity(
        fish_568_to_analyze, in_range=(min_intensity, max_intensity)
    )
    print("done!")

    fish_647_puncta_df = pd.DataFrame()
    fish_568_puncta_df = pd.DataFrame()
    for threshold in range(FISH_THRESHOLD_MIN, FISH_THRESHOLD_MAX, FISH_THRESHOLD_STEP):
        fish_647_threshold = threshold
        fish_568_threshold = threshold
        print(
            f"Thresholds set at {fish_568_threshold} (568) and {fish_647_threshold} (647)"
        )

        detections_647_at_thrs = {}
        detections_568_at_thrs = {}
        for idx, row in nuclei_props_df[nuclei_props_df["keep"]].iterrows():
            lbl = row["label"]
            print(f"--- {idx + 1} / {len(nuclei_props_df)} (lbl: {lbl}) ---")
            sl = row["slice"]

            # Find 647 blobs locations within the bbox, mask, and shift coordinates
            print(f"{ch_dict[FISH_647_CH]} channel")
            ctrs = detect_blobs(
                fish_647_to_analyze[sl[0], sl[1], sl[2]].astype("float"),
                min_sigma=2,
                max_sigma=2,
                num_sigma=1,
                z_y_x_ratio=(1 / spacing_ratio, 1, 1),
                threshold=fish_647_threshold,
            ) + (sl[0].start, sl[1].start, sl[2].start, 0, 0, 0)

            df = pd.DataFrame(
                ctrs, columns=["Z", "Y", "X", "sigma_Z", "sigma_Y", "sigma_X"]
            )

            if not df.empty:
                # Drop detections that are not inside this particular nucleus
                df.loc[:, "keep"] = False
                for idx, row in df.iterrows():
                    coo = row[["Z", "Y", "X"]].astype("uint16")
                    label = nuclei_labels[coo[0], coo[1], coo[2]]
                    if label != 0 and label == lbl:
                        df.at[idx, "keep"] = True
                df = df[df["keep"]].drop(columns=["keep"])

                # If multiple nuclei regions with same label, concatenate detections
                if lbl in detections_647_at_thrs.keys():
                    df = pd.concat([detections_647_at_thrs[lbl], df], ignore_index=True)

                # If the dataframe is not empty, assign the nucleus label
                if not df.empty:
                    df.loc[:, "nucleus"] = lbl
                    detections_647_at_thrs[lbl] = df

            print(f"Detected {len(df)} puncta")

        # Combine detections for this threshold
        detections_647_at_thrs_df = pd.concat(
            detections_647_at_thrs.values(), ignore_index=True
        )
        detections_647_at_thrs_df["thresholds"] = [
            [fish_647_threshold] for _ in range(len(detections_647_at_thrs_df))
        ]
        detections_647_at_thrs_df.loc[:, "label"] = range(
            1, len(detections_647_at_thrs_df) + 1
        )

        # Combine in overall detection dataframe
        if fish_647_puncta_df.empty:
            fish_647_puncta_df = detections_647_at_thrs_df.copy()
        else:
            # Match new detection with closest from existing puncta
            fish_647_tree = sci_spa.KDTree(
                fish_647_puncta_df.loc[:, ["Z", "Y", "X"]].to_numpy()
                * (spacing_ratio, 1, 1)
            )
            closest_detection = fish_647_tree.query(
                detections_647_at_thrs_df.loc[:, ["Z", "Y", "X"]].to_numpy()
                * (spacing_ratio, 1, 1)
            )
            detections_647_at_thrs_df.loc[:, "label"] = closest_detection[1] + 1
            for _, row in detections_647_at_thrs_df.iterrows():
                fish_647_puncta_df.loc[
                    fish_647_puncta_df["label"] == row["label"], "thresholds"
                ].iat[0] += [fish_647_threshold]

        # Create a dataframe with the detections for the current threhsolds and
        # merge with nuclei info
        props_df = pd.merge(
            detections_647_at_thrs_df.groupby("nucleus", as_index=False).size(),
            detections_647_at_thrs_df.groupby("nucleus", as_index=False).agg(list)[
                ["nucleus", "label"]
            ],
            on="nucleus",
        ).rename(
            columns={
                "nucleus": "label",
                "size": ch_dict[FISH_647_CH] + f"_cnt_{fish_647_threshold:03}",
                "label": ch_dict[FISH_647_CH] + f"_ids_{fish_647_threshold:03}",
            }
        )
        nuclei_props_df = nuclei_props_df.merge(props_df, on="label", how="left")
        ##################################

    print("hello!")
    input()
    #     # Select FISH signatures within nuclei
    #     print("Assigning FISH puncta to nuclei:...", end="", flush=True)
    #     fish_647_puncta_df.loc[:, ["keep", "nucleus"]] = False, None
    #     nuclei_props_df.loc[
    #         :, ch_dict[FISH_647_CH] + f"_cnt_{fish_647_threshold:03}"
    #     ] = 0
    #     nuclei_props_df[ch_dict[FISH_647_CH] + f"_ids_{fish_647_threshold:03}"] = [
    #         [] for _ in range(len(nuclei_props_df))
    #     ]

    #     for _, row in fish_647_puncta_df.iterrows():
    #         coo = row[["Z", "Y", "X"]].astype("uint16")
    #         label = nuclei_labels[coo[0], coo[1], coo[2]]
    #         if label != 0:
    #             nuclei_props_df.loc[
    #                 nuclei_props_df["label"] == label,
    #                 ch_dict[FISH_647_CH] + f"_cnt_{fish_647_threshold:03}",
    #             ] += 1
    #             nuclei_props_df.loc[
    #                 nuclei_props_df["label"] == label,
    #                 ch_dict[FISH_647_CH] + f"_ids_{fish_647_threshold:03}",
    #             ].values[0].append(row["label"])
    #             fish_647_puncta_df.loc[
    #                 fish_647_puncta_df["label"] == row["label"],
    #                 ["keep", "nucleus"],
    #             ] = (True, label)

    #     fish_568_puncta_df.loc[:, ["keep", "nucleus"]] = False, None
    #     nuclei_props_df.loc[
    #         :, ch_dict[FISH_568_CH] + f"_cnt_{fish_568_threshold:03}"
    #     ] = 0
    #     nuclei_props_df[ch_dict[FISH_568_CH] + f"_ids_{fish_568_threshold:03}"] = [
    #         [] for _ in range(len(nuclei_props_df))
    #     ]

    #     for _, row in fish_568_puncta_df.iterrows():
    #         coo = row[["Z", "Y", "X"]].astype("uint16")
    #         label = nuclei_labels[coo[0], coo[1], coo[2]]
    #         if label != 0:
    #             nuclei_props_df.loc[
    #                 nuclei_props_df["label"] == label,
    #                 ch_dict[FISH_568_CH] + f"_cnt_{fish_568_threshold:03}",
    #             ] += 1
    #             nuclei_props_df.loc[
    #                 nuclei_props_df["label"] == label,
    #                 ch_dict[FISH_568_CH] + f"_ids_{fish_568_threshold:03}",
    #             ].values[0].append(row["label"])
    #             fish_568_puncta_df.loc[
    #                 fish_568_puncta_df["label"] == row["label"],
    #                 ["keep", "nucleus"],
    #             ] = (True, label)

    #     print("done!")

    #     # Some stats
    #     print(f"{ch_dict[FISH_647_CH]} channel:")
    #     tmp = fish_647_puncta_df.loc[fish_647_puncta_df["keep"]]
    #     print(f"Detected {len(tmp)} puncta")
    #     print(tmp[["sigma_Z", "sigma_Y", "sigma_X"]].describe())

    #     print(f"{ch_dict[FISH_568_CH]} channel:")
    #     tmp = fish_568_puncta_df.loc[fish_568_puncta_df["keep"]]
    #     print(f"Detected {len(tmp)} puncta")
    #     print(tmp[["sigma_Z", "sigma_Y", "sigma_X"]].describe())

    #     # Visualize analyzed images and create point layers with FISH puncta within nuclei
    #     # TODO: find a way to visualize puncta for all the thresholds
    #     if visualize:
    #         viewer.add_image(
    #             fish_647_to_analyze,
    #             name=ch_dict[FISH_647_CH] + "-analized",
    #             colormap="magenta",
    #             blending="additive",
    #             scale=spacing,
    #             interpolation="nearest",
    #             visible=False,
    #         )
    #         viewer.add_image(
    #             fish_568_to_analyze,
    #             name=ch_dict[FISH_568_CH] + "-analized",
    #             colormap="magenta",
    #             blending="additive",
    #             scale=spacing,
    #             interpolation="nearest",
    #             visible=False,
    #         )

    #         viewer.add_points(
    #             fish_647_puncta_df.loc[
    #                 fish_647_puncta_df["keep"], ["Z", "Y", "X"]
    #             ].to_numpy(),
    #             name=ch_dict[FISH_647_CH] + "-puncta",
    #             size=15,
    #             symbol="ring",
    #             opacity=0.2,
    #             scale=spacing,
    #             edge_color="blue",
    #             face_color="blue",
    #             blending="additive",
    #             out_of_slice_display=True,
    #             visible=False,
    #         )

    #         viewer.add_points(
    #             fish_568_puncta_df.loc[
    #                 fish_568_puncta_df["keep"], ["Z", "Y", "X"]
    #             ].to_numpy(),
    #             name=ch_dict[FISH_568_CH] + "-puncta",
    #             size=15,
    #             symbol="ring",
    #             opacity=0.2,
    #             scale=spacing,
    #             edge_color="red",
    #             face_color="red",
    #             blending="additive",
    #             out_of_slice_display=True,
    #             visible=False,
    #         )

    #     # Plot 647 vs 538 for each nucleus
    #     top = max(
    #         nuclei_props_df[
    #             ch_dict[FISH_568_CH] + f"_cnt_{fish_568_threshold:03}"
    #         ].max(),
    #         nuclei_props_df[
    #             ch_dict[FISH_647_CH] + f"_cnt_{fish_647_threshold:03}"
    #         ].max(),
    #     )
    #     joint_plot = sbn.jointplot(
    #         data=nuclei_props_df[nuclei_props_df["keep"]],
    #         x=ch_dict[FISH_568_CH] + f"_cnt_{fish_568_threshold:03}",
    #         y=ch_dict[FISH_647_CH] + f"_cnt_{fish_647_threshold:03}",
    #         kind="hist",
    #         ratio=3,
    #         xlim=(0, top),
    #         ylim=(0, top),
    #         marginal_ticks=True,
    #         cbar=True,
    #         joint_kws={"binwidth": 1},
    #         marginal_kws={"binwidth": 1},
    #     )

    #     # Move the colorbar to the right while keeping the plot square
    #     plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1)
    #     pos_joint_ax = joint_plot.ax_joint.get_position()
    #     pos_marg_x_ax = joint_plot.ax_marg_x.get_position()
    #     joint_plot.ax_joint.set_position(
    #         [pos_joint_ax.x0, pos_joint_ax.y0, pos_marg_x_ax.width, pos_joint_ax.height]
    #     )
    #     joint_plot.fig.axes[-1].set_position(
    #         [0.83, pos_joint_ax.y0, 0.07, pos_joint_ax.height]
    #     )
    #     # Enforce same limits and ticks on both axes
    #     joint_plot.ax_joint.set_yticks(joint_plot.ax_joint.get_xticks())
    #     joint_plot.ax_joint.set_xticks(joint_plot.ax_joint.get_yticks())
    #     joint_plot.ax_joint.set_xlim(0, joint_plot.ax_joint.get_xlim()[1])
    #     joint_plot.ax_joint.set_ylim(0, joint_plot.ax_joint.get_ylim()[1])
    #     # Add title and axes labels
    #     file = osp.split(filename)
    #     plt.suptitle(file[-1])
    #     plt.xlabel(f"{ch_dict[FISH_568_CH]} count (thresh: {fish_568_threshold})")
    #     plt.ylabel(f"{ch_dict[FISH_647_CH]} count (thresh: {fish_647_threshold})")

    #     # Evaluate which nuclei have a particular FISH count
    #     nuclei_count_df = (
    #         nuclei_props_df.groupby(
    #             [
    #                 ch_dict[FISH_568_CH] + f"_cnt_{fish_568_threshold:03}",
    #                 ch_dict[FISH_647_CH] + f"_cnt_{fish_647_threshold:03}",
    #             ]
    #         )
    #         .agg(list)[["label"]]
    #         .reset_index()
    #     )
    #     print(f"Nuclei with given FISH counts:\n{nuclei_count_df}")

    #     # Save figure
    #     root_dir = build_path(filename)
    #     if not osp.isdir(root_dir):
    #         os.makedirs(root_dir)
    #     plt.savefig(
    #         build_path(
    #             filename,
    #             f"-{ch_dict[NUCLEI_CH]}-FISH-({fish_568_threshold:03}, {fish_647_threshold:03})-plot.pdf",
    #         ),
    #         bbox_inches="tight",
    #     )

    #     # Save the dataframes
    #     fish_647_puncta_df.to_json(
    #         build_path(
    #             filename,
    #             f"-{ch_dict[FISH_647_CH]}-FISH-({fish_647_threshold:03})-df.json",
    #         )
    #     )
    #     fish_568_puncta_df.to_json(
    #         build_path(
    #             filename,
    #             f"-{ch_dict[FISH_568_CH]}-FISH-({fish_568_threshold:03})-df.json",
    #         )
    #     )
    #     nuclei_count_df.to_json(
    #         build_path(
    #             filename,
    #             f"-FISH_count_per_nuclei-({fish_568_threshold:03}, {fish_647_threshold:03})-df.json",
    #         )
    #     )
    #     print("done!")

    #     if visualize:
    #         plt.show(block=False)
    #         napari.run()

    # # Save nuclei dataframe
    # print(
    #     "Saving the nuclei's and puncta's property dataframes...",
    #     end="",
    #     flush=True,
    # )
    # nuclei_props_df.to_json(
    #     build_path(
    #         filename,
    #         f"-{ch_dict[NUCLEI_CH]}-FISH-({FISH_THRESHOLD_MIN}-{FISH_THRESHOLD_MAX}-{FISH_THRESHOLD_STEP})-df.json",
    #     )
    # )


##############################################################################################
# # Bounding surfaces for thresholded nuclei
# print("Evaluating thresholded nuclei's boundaries...", end="", flush=True)
# nuclei_mask_boundaries = ski_mor.dilation(
#     nuclei_mask, footprint=ski_mor.ball(1)) ^ nuclei_mask
# viewer.add_image(
#     nuclei_mask_boundaries,
#     opacity=0.5,
#     scale=spacing,
#     colormap="red",
#     blending="additive",
#     interpolation="nearest",
#     visible=False
# )
# print("done!")

# # TODO: ideas to try to detect center of nuclei:
# #       - Blurr segmented, find max, watershed with Vornoi boundaries:
# #         https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/Segmentation_3D.ipynb
# #         https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/voronoi_otsu_labeling.ipynb)
# #       - Gaussian blurr, then 3D Frangi to detect blobness then max of Frangi to detect nuclei
# #         centers and then use those for watershed.
# #       - Autocorrelation: cells are about the same size and autocorrelation should peak when
# #         they overlap. Or use one cell as template and template matching. If we can find the
# #         centers, we can then use watershed to identify individual cells.
# #       - Consider a z-by-z segmentation with prior given from neighboring slices.
# #       - Idea for automated classification: after detecting a few, use a 3x3 block with some classic classification
# #         method: KNN, SVM, etc. to classify the rest of the image. Use both the original image and the median-filtered
# #         one to generate the features 3x3 from both. Also other features could include gradient and laplacian of the
# #         image.
# #       - You really have to try Random Forest of Extra Tree. Think about a reasonable set of features to use that might
# #         help identify inside and boundaries. Also check the libraries that are available in Python to extract features
# #         "Deep learning meets radiomics for end-to-end brain tumor mri analysis" (W. Ponikiewski)
# #       - Check ICIP2022 paper #1147 (and the following on in the panel - Greek guy)
# #
# # # Evaluate and store image properties based on nuclei components
# # for ch in range(len(channels)):
# #     comp_props_df = pd.DataFrame(
# #         ski_mea.regionprops_table(
# #             nuclei_comp,
# #             intensity_image=data[ch, ...],
# #             properties=('label', 'intensity_mean')
# #         )
# #     )
# #     nuclei_props_df = nuclei_props_df.merge(comp_props_df, on='label')
# #     nuclei_props_df = nuclei_props_df.rename(columns={'intensity_mean': 'intensity_mean_ch{}'.format(channels[ch])})

# # Evaluate nuclei's volume statistics and use to select individual nuclei
# # NOTE: This only really works, when it works, with the large median filter in x and y.
# nuclei_min_vol, nuclei_med_vol, nuclei_max_vol = nuclei_props_df['area'].quantile([
#     0.25, 0.5, 0.75])
# nuclei_mean = nuclei_props_df['area'].mean()
# nuclei_std = nuclei_props_df['area'].std()
# print(" - Volumes quantiles: {}, {}, {}".format(
#     nuclei_min_vol,
#     nuclei_med_vol,
#     nuclei_max_vol
# )
# )
# print(" - Selected volume range: {} - {}".format(
#     nuclei_min_vol,
#     nuclei_max_vol)
# )

# # Show statistics
# sbn.displot(
#     nuclei_props_df,
#     x='area',
#     kde=True,
#     rug=True
# )
# plt.axvline(nuclei_med_vol, color="red")
# plt.axvline(nuclei_min_vol, color="red")
# plt.axvline(nuclei_max_vol, color="red")
# plt.axvline(nuclei_mean, color="green")
# plt.axvline(nuclei_mean + nuclei_std, color="green")
# plt.axvline(nuclei_mean - nuclei_std, color="green")

# plt.figure()
# sbn.boxplot(
#     nuclei_props_df,
#     x='area',
#     notch=True,
#     showcaps=False
# )

# # Remove large and small components based on percentiles
# print(" - Removing unwanted nuclei connected components...", end="", flush=True)
# nuclei_props_df.loc[:, 'keep'] = False
# qry = 'area > ' + str(nuclei_min_vol) + ' & area < ' + str(nuclei_max_vol)
# mask = nuclei_props_df.query(qry).index
# nuclei_props_df.loc[mask, 'keep'] = True

# # Generate the cleaned nuclei bool mask and the labeled components
# nuclei_mask_cleaned = nuclei_mask.copy()
# for _, row in nuclei_props_df[nuclei_props_df['keep'] == False].iterrows():
#     nuclei_mask_cleaned[nuclei_comp == row['label']] = False

# viewer.add_image(
#     nuclei_mask_cleaned,
#     scale=spacing,
#     opacity=0.5,
#     colormap="blue",
#     blending="additive",
#     interpolation="nearest",
#     visible=False
# )
# print("done!")

# print("Nuclei's dataframe:\n", nuclei_props_df)

# # Bounding surfaces for uniquely detected nuclei
# print("Evaluating uniquely detected nuclei's boundaries...", end="", flush=True)
# nuclei_mask_cleaned_boundaries = ski_mor.dilation(
#     nuclei_mask_cleaned, footprint=ski_mor.ball(1)) ^ nuclei_mask_cleaned
# viewer.add_image(
#     nuclei_mask_cleaned_boundaries,
#     opacity=0.5,
#     scale=spacing,
#     colormap="red",
#     blending="additive",
#     interpolation="nearest",
#     visible=False
# )
# print("done!")

# # Mask original nuclei channel
# print("Masking nuclei channel...", end="", flush=True)
# nuclei_masked = data[NUCLEI_CH].copy()
# nuclei_masked[nuclei_mask_cleaned == False] = 0
# viewer.add_image(
#     nuclei_masked,
#     scale=spacing,
#     opacity=1,
#     colormap="green",
#     blending="additive",
#     interpolation="nearest",
#     visible=False
# )
# print("done!")

# # Mask original labels
# print("Masking nuclei connected components...", end="", flush=True)
# # nuclei_labels = ski_mea.label(nuclei_mask_cleaned)
# nuclei_labels = nuclei_comp.copy()
# nuclei_labels[nuclei_mask_cleaned == False] = 0
# viewer.add_labels(
#     nuclei_labels,
#     scale=spacing,
#     blending="additive",
#     visible=False
# )
# print("done!")

# # TODO: Software to check:
# #        - CellProfiler (https://cellprofiler.org/)
# #        - CellSegm (https://scfbm.biomedcentral.com/articles/10.1186/1751-0473-8-16)
# #        - SMMF algorithm (https://ieeexplore.ieee.org/document/4671118)
# #        - Imaris methods
# #        - 3D UNET, nnUNET
# #        - pyradiomics (https://pyradiomics.readthedocs.io/en/latest/index.html) for feature extraction


from glob import glob

if __name__ == "__main__":
    analyze_image(visualize=False)

    # files = [
    #     g
    #     for p in [
    #         "/Volumes/GoogleDrive/Shared drives/MarkD/FISH analysis/FISH Nov 2022/20221101_cad87A-568_dpr17-647_2(DAPI-405).czi",
    #     ]
    #     for g in glob(p)
    # ]
    # files.sort()
    # for f in files:
    #     analyze_image(
    #         f,
    #         visualize=False,
    #     )
