import argparse
import os
import os.path as osp
from tkinter import filedialog

import colorama
from colorama import Fore, Style
import matplotlib.pyplot as plt
import napari
import numpy as np
import pandas as pd
import scipy.ndimage as sci_ndi
import scipy.spatial as sci_spa
import seaborn as sbn
import skimage.exposure as ski_exp
import skimage.feature as ski_fea
import skimage.filters as ski_fil
import skimage.filters.rank as ski_fil_ran
import skimage.measure as ski_mea
import skimage.morphology as ski_mor
import skimage.segmentation as ski_seg
from magicgui import magicgui

import os_utils
from image import Image
from metadata import Metadata

colorama.init()

#! IMPORTANT FOR CONSISTENCY: add a flag so that if there is an exception with a
#! file, all the following steps are re-evaluated instead of using stored values
#! Check the "refresh" argument of the analyze_image() function.


def eval_stats(data, mask=None):
    masked = data[mask] if mask is not None else data
    return masked.min(), np.median(masked), masked.max()


def contrast_stretch(data, in_range="image", mask=None, ch_id=None):
    print(
        f"Contrast stretching {Fore.GREEN}{Style.BRIGHT}{ch_id}{Style.RESET_ALL}... ",
        end="",
        flush=True,
    )
    input_min, input_median, input_max = eval_stats(data, mask)
    data = ski_exp.rescale_intensity(data, in_range)
    scaled_min, scaled_median, scaled_max = eval_stats(data, mask)
    print(
        f"{Style.BRIGHT}[{input_min}, {input_median}, {input_max}] => {in_range} => [{scaled_min}, {scaled_median}, {scaled_max}]{Style.RESET_ALL}... done!"
    )
    return data


def filter(
    data,
    type="median",
    footprint=None,
    sigma=(2, 10, 10),
    mode="nearest",
    cval=0,
    filename_root=None,
    ch_id=None,
):
    filtered = data.copy()

    if type not in ["median", "gaussian", "closing", "maximum"]:
        raise ValueError("WARNING: the specified mode is not available.")

    if footprint is None:
        footprint = ski_mor.ball(1)
    footprint_dim = footprint.shape

    print(
        f"Applying {Fore.BLUE}{type}{Style.RESET_ALL} filter to {Fore.GREEN}{Style.BRIGHT}{ch_id}{Style.RESET_ALL}... ",
        end="",
        flush=True,
    )

    if filename_root is None:
        if type == "closing":
            filtered = ski_mor.closing(data, footprint=footprint)
        elif type == "gaussian":
            filtered = sci_ndi.gaussian_filter(data, mode=mode, cval=cval, sigma=sigma)
        elif type == "maximum":
            filtered = ski_fil_ran.maximum(data, footprint=footprint)
        elif type == "median":
            filtered = sci_ndi.median_filter(
                data, mode=mode, cval=cval, footprint=footprint
            )
    elif ch_id is None:
        raise ValueError("WARNING: a ch_id should be provided to identify the channel.")
    else:
        if type == "closing":
            file = os_utils.build_path(
                filename_root,
                f"-{ch_id}-clos-{footprint_dim}.npy",
            )
        elif type == "gaussian":
            file = os_utils.build_path(
                filename_root,
                f"-{ch_id}-den-gaus-{tuple(np.round(sigma, decimals=2))}.npy",
            )
        elif type == "maximum":
            file = os_utils.build_path(
                filename_root,
                f"-{ch_id}-max-{footprint_dim}.npy",
            )
        elif type == "median":
            file = os_utils.build_path(
                filename_root, f"-{ch_id}-den-med-{footprint_dim}.npy"
            )
        try:
            filtered = os_utils.load(file)
        except (FileNotFoundError, ValueError):
            if type == "closing":
                filtered = ski_mor.closing(data, footprint=footprint)
            elif type == "gaussian":
                filtered = sci_ndi.gaussian_filter(
                    data, mode=mode, cval=cval, sigma=sigma
                )
            elif type == "maximum":
                filtered = ski_fil_ran.maximum(data, footprint=footprint)
            elif type == "median":
                filtered = sci_ndi.median_filter(
                    data, mode=mode, cval=cval, footprint=footprint
                )
            try:
                root_dir = os_utils.build_path(filename_root)
                if not osp.isdir(root_dir):
                    os.makedirs(root_dir)
                os_utils.save(file, filtered)
            except:
                print("WARNING: error saving the file.")

    print("done!")

    return filtered


def remove_floor(data, sigma=(20, 100, 100), filename_root=None, ch_id=None, mask=None):
    print(f"Removing floor from {Fore.GREEN}{Style.BRIGHT}{ch_id}{Style.RESET_ALL}:")
    input_min, input_median, input_max = eval_stats(data, mask=mask)
    noise_floor = filter(
        data, type="gaussian", sigma=sigma, filename_root=filename_root, ch_id=ch_id
    )
    defloored = data.astype("int16") - noise_floor.astype("int16")
    defloored = np.maximum(defloored, 0).astype("uint8")
    defloored_min, defloored_median, defloored_max = eval_stats(defloored, mask=mask)
    print(
        f"{Style.BRIGHT}[{input_min}, {input_median}, {input_max}] => [{defloored_min}, {defloored_median}, {defloored_max}]{Style.RESET_ALL}"
    )

    return defloored


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
    print("Detecting blobs' centers... ", end="", flush=True)
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
    elif ch_id is None:
        print(
            "WARNING: a ch_id should be provided to identify the channel. Blobs were not detected."
        )
    else:
        file = os_utils.build_path(filename_root, f"-{ch_id}-blb.npy")
        try:
            blobs_ctrs = os_utils.load(file)
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
                root_dir = os_utils.build_path(filename_root)
                if not osp.isdir(root_dir):
                    os.makedirs(root_dir)
                os_utils.save(file, blobs_ctrs)
            except:
                print("WARNING: error saving the file.")
    print("done!")

    return blobs_ctrs


def _evaluate_voronoi(mask, markers, spacing=(1, 1, 1)):
    markers_tree = sci_spa.KDTree(markers * spacing)
    mask_idx = mask.nonzero()
    mask_idx_array = np.vstack(mask_idx).T
    closest_marker = markers_tree.query(mask_idx_array * spacing)
    labels = np.zeros_like(mask, dtype="uint16")
    # The +1 is to start the labels at 1 instead of 0
    labels[mask_idx[0], mask_idx[1], mask_idx[2]] = closest_marker[1] + 1

    return labels


def evaluate_voronoi(mask, markers, spacing=(1, 1, 1), filename_root=None, ch_id=None):
    # Find the equivalent to Voronoi regions in the provided mask based on the
    # provided markers
    # Coordinates are normalized to the physical size using 'spacing'
    print(
        f"Identifying regions within {Fore.GREEN}{Style.BRIGHT}{ch_id}{Style.RESET_ALL}'s mask with Vornoi... ",
        end="",
        flush=True,
    )
    if filename_root is None:
        labels = _evaluate_voronoi(mask, markers, spacing)
    elif ch_id is None:
        raise ValueError(
            "A ch_id should be provided to identify the channel. Vornoi regions were not evaluated."
        )
    else:
        file = os_utils.build_path(filename_root, f"-{ch_id}-voronoi.npy")
        try:
            labels = os_utils.load(file)
        except (FileNotFoundError, ValueError):
            labels = _evaluate_voronoi(mask, markers, spacing)
            try:
                root_dir = os_utils.build_path(filename_root)
                if not osp.isdir(root_dir):
                    os.makedirs(root_dir)
                os_utils.save(file, labels)
            except:
                print("WARNING: error saving the file.")

    print("done!")

    return labels


def _evaluate_watershed(mask, markers):
    # Create the markers
    mask_img = np.zeros_like(mask, dtype="uint16")
    mask_img[
        markers[:, 0].astype("uint16"),
        markers[:, 1].astype("uint16"),
        markers[:, 2].astype("uint16"),
    ] = range(1, 1 + len(markers))

    # Remove markers that are not contained in the nuclei's mask
    mask_img_clean = mask_img & 65535 * mask.astype("uint16")

    return ski_seg.watershed(mask, mask_img_clean, mask=mask, compactness=0).astype(
        "uint16"
    )


def evaluate_watershed(mask, markers, filename_root=None, ch_id=None):
    # Use watershed to identify regions connected to the nuclei centers
    print(
        f"Identifying regions within {Fore.GREEN}{Style.BRIGHT}{ch_id}{Style.RESET_ALL}'s mask with watershed... ",
        end="",
        flush=True,
    )

    if filename_root is None:
        labels = _evaluate_watershed(mask, markers)
    elif ch_id is None:
        raise ValueError(
            "A ch_id should be provided to identify the channel. Watershed regions were not evaluated."
        )
    else:
        file = os_utils.build_path(filename_root, f"-{ch_id}-watershed.npy")
        try:
            labels = os_utils.load(file)
        except (FileNotFoundError, ValueError):
            labels = _evaluate_watershed(mask, markers)
            try:
                root_dir = os_utils.build_path(filename_root)
                if not osp.isdir(root_dir):
                    os.makedirs(root_dir)
                os_utils.save(file, labels)
            except:
                print("WARNING: error saving the file.")

    print("done!")

    return labels


def _get_fish_puncta(
    fish_channel,
    ch_id,
    nuclei_labels,
    nuclei_props_df,
    spacing_ratio=(1, 1, 1),
    thresh_min=5,
    thresh_max=251,
    thresh_step=5,
):
    fish_puncta_df = pd.DataFrame()
    props_df = pd.DataFrame()
    for threshold in range(thresh_min, thresh_max, thresh_step):
        detections_at_thrs = {}
        for n_idx, n_row in nuclei_props_df[nuclei_props_df["keep"]].iterrows():
            lbl = n_row["label"]
            print(
                f"--- Ch: {ch_id} - Thrs: {threshold} - {n_idx + 1:2} / {len(nuclei_props_df):2} (lbl: {lbl:2}) ---"
            )
            sl = n_row["slice"]

            # Find 647 blobs locations within the bbox, mask, and shift coordinates
            # TODO: consider using a range of sigmas and then Frangi to preserve the most blobby
            ctrs = detect_blobs(
                fish_channel[sl[0], sl[1], sl[2]].astype("float"),
                min_sigma=2,
                max_sigma=2,
                num_sigma=1,
                z_y_x_ratio=(1 / spacing_ratio, 1, 1),
                threshold=threshold,
            ) + (sl[0].start, sl[1].start, sl[2].start, 0, 0, 0)

            df = pd.DataFrame(
                ctrs, columns=["Z", "Y", "X", "sigma_Z", "sigma_Y", "sigma_X"]
            )

            if not df.empty:
                # Drop detections that are not inside this particular nucleus
                df.loc[:, "keep"] = False
                for d_idx, d_row in df.iterrows():
                    coo = d_row[["Z", "Y", "X"]].astype("uint16")
                    label = nuclei_labels[coo[0], coo[1], coo[2]]
                    if label != 0 and label == lbl:
                        df.at[d_idx, "keep"] = True
                df = df[df["keep"]].drop(columns=["keep"])

                # If multiple nuclei regions with same label, concatenate detections
                if lbl in detections_at_thrs:
                    df = pd.concat([detections_at_thrs[lbl], df], ignore_index=True)

            # If the dataframe is not empty, assign the nucleus label
            if not df.empty:
                df.loc[:, "nucleus"] = lbl
                detections_at_thrs[lbl] = df

            print(
                f"Detected {Fore.BLUE}{len(df)} puncta{Style.RESET_ALL} ({len(df) / n_row['area']:.4f} puncta/pixel)"
            )

        # Assumption: if there are no detections at a given threshold, there
        # will be none at higher thresholds.
        # TODO: Modify to fill the remaining thresholds with 0 cnt and [] ids
        if not detections_at_thrs:
            break

        # Combine detections for this threshold
        detections_at_thrs_df = pd.concat(
            detections_at_thrs.values(), ignore_index=True
        )
        detections_at_thrs_df["thresholds"] = [
            [threshold] for _ in range(len(detections_at_thrs_df))
        ]
        detections_at_thrs_df.loc[:, "label"] = range(1, len(detections_at_thrs_df) + 1)

        # Combine in overall detection dataframe
        if fish_puncta_df.empty:
            fish_puncta_df = detections_at_thrs_df.copy()
        else:
            # Match new detection with closest from existing puncta
            fish_tree = sci_spa.KDTree(
                fish_puncta_df.loc[:, ["Z", "Y", "X"]].to_numpy()
                * (spacing_ratio, 1, 1)
            )
            closest_detection = fish_tree.query(
                detections_at_thrs_df.loc[:, ["Z", "Y", "X"]].to_numpy()
                * (spacing_ratio, 1, 1)
            )
            detections_at_thrs_df.loc[:, "label"] = closest_detection[1] + 1
            for _, n_row in detections_at_thrs_df.iterrows():
                fish_puncta_df.loc[
                    fish_puncta_df["label"] == n_row["label"], "thresholds"
                ].iat[0] += [threshold]

        # Create a dataframe with the detections for the current thresholds
        # and merge with nuclei info
        df = pd.merge(
            detections_at_thrs_df.groupby("nucleus", as_index=False).size(),
            detections_at_thrs_df.groupby("nucleus", as_index=False).agg(list)[
                ["nucleus", "label"]
            ],
            on="nucleus",
        ).rename(
            columns={
                "nucleus": "label",
                "size": ch_id + f"_cnt_{threshold:03}",
                "label": ch_id + f"_ids_{threshold:03}",
            }
        )
        props_df = (
            df.copy() if props_df.empty else props_df.merge(df, on="label", how="left")
        )
    # Fill missing counts with zeros and missing ids with empty lists
    filt = props_df.filter(regex="cnt")
    props_df[filt.columns] = filt.fillna(0)
    filt = props_df.filter(regex="ids")
    props_df[filt.columns] = filt.fillna(props_df.notna().applymap(lambda x: x or []))

    return fish_puncta_df, props_df


def get_fish_puncta(
    fish_channel,
    nuclei_labels,
    nuclei_props_df,
    spacing_ratio=(1, 1, 1),
    thresh_min=5,
    thresh_max=251,
    thresh_step=5,
    filename_root=None,
    ch_id=None,
):
    if ch_id is None:
        raise ValueError(
            "A ch_id should be provided to identify the channel. FISH puncta are not detected."
        )
    elif filename_root is None:
        print(f"Looking for FISH signatures in channel {ch_id}:")
        fish_puncta_df, props_df = _get_fish_puncta(
            fish_channel,
            ch_id,
            nuclei_labels,
            nuclei_props_df,
            spacing_ratio,
            thresh_min,
            thresh_max,
            thresh_step,
        )
    else:
        file_puncta = os_utils.build_path(
            filename_root,
            f"-{ch_id}-FISH-({thresh_min}-{thresh_max}-{thresh_step})-df",
        )
        file_props = os_utils.build_path(
            filename_root,
            f"-{ch_id}-FISH-({thresh_min}-{thresh_max}-{thresh_step})-props-df",
        )
        try:
            fish_puncta_df = pd.read_json(file_puncta + ".json")
            props_df = pd.read_json(file_props + ".json")
        except FileNotFoundError:
            print(f"Looking for FISH signatures in channel {ch_id}:")
            fish_puncta_df, props_df = _get_fish_puncta(
                fish_channel,
                ch_id,
                nuclei_labels,
                nuclei_props_df,
                spacing_ratio,
                thresh_min,
                thresh_max,
                thresh_step,
            )
            try:
                root_dir = os_utils.build_path(filename_root)
                if not osp.isdir(root_dir):
                    os.makedirs(root_dir)
                fish_puncta_df.to_json(file_puncta + ".json")
                fish_puncta_df.to_csv(file_puncta + ".csv")
                props_df.to_json(file_props + ".json")
                props_df.to_csv(file_props + ".csv")
            except:
                print("WARNING: error saving the file.")

    return fish_puncta_df, props_df


def analyze_image(
    filename=None,
    visualize=False,
    channels=4,
    refresh=False,
    metadata=False,
    fish_range=None,
    no_cyto=False,
):
    # Ask user to choose a file
    print(f"\n{Fore.RED}{Style.BRIGHT}--- Starting new analysis ---{Style.RESET_ALL}")
    if filename is None:
        filename = filedialog.askopenfilename()
        if filename == "":
            raise ValueError("A .czi file should be provided for analysis.")

    # Load image and extract data and metadata
    print(f"Loading file {Style.BRIGHT}{Fore.GREEN}{filename}{Style.RESET_ALL}")
    image = Image(filename)
    image.load_image()
    data = image.get_data()

    # --- Development only: shrink data ---
    # dds = [np.floor(d//4).astype('uint16') for d in data.shape]
    # dde = [np.ceil(d - d//4).astype('uint16') for d in data.shape]
    # data = data[:, dds[1]:dde[1], dds[2]:dde[2], dds[3]:dde[3]]
    # -------------------------------------

    # Check if the number of required channels corresponds to the channels in
    # the image.
    if channels != image.channels_no:
        raise ValueError(
            f"Number of required channels ({channels}) does not correspond to channels in the image ({image.channels_no})."
        )

    # Specify channels
    #! This works with our data only
    # TODO: since it can change from image to image, do it through metadata
    if channels == 4:
        if no_cyto:
            raise ValueError(
                f"Configuration of channels ({channels}) and no_cyto ({no_cyto}) not supported."
            )
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
    elif channels == 3:
        if no_cyto:
            NUCLEI_CH = 0
            FISH_647_CH = 1
            FISH_568_CH = 2
            ch_dict = {
                NUCLEI_CH: "Nuclei",
                FISH_647_CH: "FISH_647",
                FISH_568_CH: "FISH_568",
            }
        else:
            NUCLEI_CH = 0
            FISH_568_CH = 1
            CYTO_CH = 2
            ch_dict = {
                NUCLEI_CH: "Nuclei",
                FISH_568_CH: "FISH_568",
                CYTO_CH: "Cytoplasm",
            }
    else:
        raise ValueError(f"Number of channels ({channels}) not allowed.")

    # Specify thresholds for FISH detection
    FISH_THRESHOLD_MIN = 5
    FISH_THRESHOLD_MAX = 251
    FISH_THRESHOLD_STEP = 5

    # Gather and report image information
    pixel_sizes = image.scaling
    spacing = pixel_sizes
    spacing_ratio = int(np.ceil(spacing[0] / spacing[1]))
    contrast = [[np.min(data[ch]), np.max(data[ch])] for ch in range(data.shape[0])]
    print(
        f"{Style.BRIGHT}{Fore.BLUE}########## Original data info: ##########{Style.RESET_ALL}"
    )
    print(f"{Style.BRIGHT}Image shape (CH, Z, Y, X):{Style.RESET_ALL}")
    print(f"  {data.shape}")
    print(f"{Style.BRIGHT}Channels:{Style.RESET_ALL}")
    for (k, v), c in zip(image.channels_meta.items(), list(ch_dict.values())):
        print(f"  {k}: {c:9} <=> {v}")
    print(f"{Style.BRIGHT}Pixel sizes (Z, Y, X):{Style.RESET_ALL}")
    print(f"  {spacing}")
    print(f"{Style.BRIGHT}Spacing ratio (Z / X or Y):")
    print(f"  {spacing_ratio}")
    print(f"{Style.BRIGHT}Data type:{Style.RESET_ALL}")
    for k, v in image.type_meta.items():
        print(f"  {k} => {v}")
    print(f"{Style.BRIGHT}Data ranges:{Style.RESET_ALL}")
    for (k, v), r in zip(list(ch_dict.items()), contrast):
        print(f"  {k}: {v:9} => {r}")

    print(
        f"{Style.BRIGHT}{Fore.BLUE}#########################################{Style.RESET_ALL}"
    )

    # Stop if we only want the metadata
    if metadata:
        return

    # Contrast stretch nuclei and cyto
    for ch in (NUCLEI_CH, CYTO_CH):
        data[ch] = contrast_stretch(data[ch], ch_id=ch_dict[ch])

    # Contrast stretch the FISH channels
    #! This works with our data only
    if channels == 3:
        ch = FISH_568_CH
        if fish_range is None:
            data[ch] = contrast_stretch(data[ch], ch_id=ch_dict[ch])
        else:
            data[ch] = contrast_stretch(
                data[ch], in_range=(fish_range[0], fish_range[1]), ch_id=ch_dict[ch]
            )
    elif channels == 4:
        min_intensity, max_intensity = np.inf, -np.inf
        min_intensity = min(
            min_intensity, data[FISH_568_CH].min(), data[FISH_647_CH].min()
        )
        max_intensity = max(
            max_intensity, data[FISH_568_CH].max(), data[FISH_647_CH].max()
        )
        for ch in (FISH_647_CH, FISH_568_CH):
            if fish_range is None:
                data[ch] = contrast_stretch(
                    data[ch], in_range=(min_intensity, max_intensity), ch_id=ch_dict[ch]
                )
            else:
                data[ch] = contrast_stretch(
                    data[ch],
                    in_range=(fish_range[0], fish_range[1]),
                    ch_id=ch_dict[ch],
                )

    # If needed, convert to uint8
    if data.dtype != "uint8":
        for ch in range(data.shape[0]):
            print(
                f"Converting {Fore.GREEN}{Style.BRIGHT}{ch_dict[ch]}{Style.RESET_ALL} from uint16 to uint8...",
                end="",
                flush=True,
            )
            input_min, input_median, input_max = eval_stats(data[ch])
            data[ch] = data[ch] // 256
            converted_min, converted_median, converted_max = eval_stats(data[ch])
            print(
                f" {Style.BRIGHT}[{input_min}, {input_median}, {input_max}] => [{converted_min}, {converted_median}, {converted_max}]{Style.RESET_ALL}... done!"
            )
        data = data.astype("uint8")

    # Show original data
    if visualize:
        # Show pre-processed data
        viewer = napari.Viewer(title=osp.split(filename)[1], ndisplay=3)
        viewer.add_image(
            data,
            channel_axis=0,
            name=[ch + "-orig" for ch in ch_dict.values()],
            colormap="green",
            blending="additive",
            scale=spacing,
            depiction="volume",
            interpolation="nearest",
            visible=False,
        )

    # Remove floor from each channel
    for ch in range(data.shape[0]):
        data[ch] = remove_floor(
            data[ch],
            sigma=100 * np.array((1 / spacing_ratio, 1, 1)),
            filename_root=filename,
            ch_id=ch_dict[ch],
        )

    # Contrast stretch nuclei and cyto
    for ch in (NUCLEI_CH, CYTO_CH):
        data[ch] = contrast_stretch(data[ch], ch_id=ch_dict[ch])

    # Contrast stretch the FISH channels
    #! This works with our data only
    if channels == 3:
        ch = FISH_568_CH
        data[ch] = contrast_stretch(data[ch], ch_id=ch_dict[ch])
    elif channels == 4:
        min_intensity, max_intensity = np.inf, -np.inf
        min_intensity = min(
            min_intensity, data[FISH_568_CH].min(), data[FISH_647_CH].min()
        )
        max_intensity = max(
            max_intensity, data[FISH_568_CH].max(), data[FISH_647_CH].max()
        )
        for ch in (FISH_647_CH, FISH_568_CH):
            data[ch] = contrast_stretch(
                data[ch], in_range=(min_intensity, max_intensity), ch_id=ch_dict[ch]
            )
    if visualize:
        # Show pre-processed data
        viewer.add_image(
            data,
            channel_axis=0,
            name=[ch + "-pre" for ch in ch_dict.values()],
            colormap="green",
            blending="additive",
            scale=spacing,
            depiction="volume",
            interpolation="nearest",
            visible=False,
        )

    # Apply median filter to denoise the nuclei channel
    print("Denoising nuclei's channel:")
    nuclei_den = filter(
        data[NUCLEI_CH],
        footprint=ski_mor.ball(7)[3::4],
        filename_root=filename,
        ch_id=ch_dict[NUCLEI_CH],
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
    print("Thresholding nuclei... ", end="", flush=True)
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

    # TODO: with the current threshold, a lot of centers are found. It might be
    #       worth changing it. You should keep in mind that the thresholds are
    #       affected by the contrast in the image. If we change how we stretch
    #       the image, it will change the required threshold.

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

    print(
        f"Detected {Style.BRIGHT}{len(nuclei_ctrs)}{Style.RESET_ALL} centers:\n{nuclei_ctrs}"
    )

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
    nuclei_labels_watershed = evaluate_watershed(
        nuclei_mask,
        nuclei_ctrs[:, :3],
        filename_root=filename,
        ch_id=ch_dict[NUCLEI_CH],
    )
    if visualize:
        viewer.add_labels(
            nuclei_labels_watershed,
            name=ch_dict[NUCLEI_CH] + "-ws-lbls",
            scale=spacing,
            blending="additive",
            visible=False,
        )

    # Final labeled nuclei mask (no cytoplasm)
    print("Combining Vornoi and watershed nuclei's labeling... ", end="", flush=True)
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

    #! This works with our data only
    # TODO: The following should be done only if the cytoplasm channel exist.
    # TODO: Use metadata?
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
    print("Thresholding cytoplasm... ", end="", flush=True)
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
    print("Inverting cytoplasm mask... ", end="", flush=True)
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
    if visualize:
        viewer.add_labels(
            cyto_inv_labels_voronoi,
            name=ch_dict[CYTO_CH] + "-inv-vor-lbls",
            scale=spacing,
            blending="additive",
            visible=False,
        )

    # Use watershed to identify regions connected to the nuclei centers
    cyto_inv_labels_watershed = evaluate_watershed(
        cyto_mask_inv,
        nuclei_ctrs[:, :3],
        filename_root=filename,
        ch_id=ch_dict[CYTO_CH],
    )
    if visualize:
        viewer.add_labels(
            cyto_inv_labels_watershed,
            name=ch_dict[CYTO_CH] + "-inv-ws-lbls",
            scale=spacing,
            blending="additive",
            visible=False,
        )

    # Final labeled nuclei mask (with cytoplasm)
    print(
        "Combining nuclei's labeling with inverse cytoplasm labeling... ",
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
    # overall is found. At each iteration we might need to Voronoi again since we
    # are modifying the mask.
    ############################################################################

    ################################################################################
    # Hack to be improved: Dilate preserved nuclei labels to identify nearby puncta
    print("Dilate preserved nuclei to include part of the surrounding cytoplasm:")
    nuclei_labels = filter(
        nuclei_labels,
        type="maximum",
        footprint=ski_mor.ball(5)[2::3],
        # footprint=ski_mor.ball(7)[3::4],
        # footprint=ski_mor.ball(9)[1::4],
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
    nuclei_labels_mask = nuclei_labels > 0
    ################################################################################

    # Evaluate potential nuclei properties
    # TODO: consider adding the detected centers and sigmas
    print("Evaluating potential nuclei properties from mask... ", end="", flush=True)
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
    print("Identifying FISH puncta's centers within nuclei...")
    # To minimize difference between channels, we are pre-building the images
    # with just the region that we will analyze and then contrast stretch
    # globally. This should normalize the puncta intensity among channels as
    # well as within the channels. Contrast stretching individual nuclei
    # area would be equivalent to use non uniform detection thresholds.

    # Extract the subset of values from teh FISH channels within the identifies
    # nuclei bounding boxes and calculate extremes
    #! This works with our data only
    if channels == 4:
        sub = data[[FISH_568_CH, FISH_647_CH]][np.stack([nuclei_labels_mask] * 2)]
    else:
        sub = data[FISH_568_CH][nuclei_labels_mask]

    min_intensity = sub.min()
    max_intensity = sub.max()

    # Contrast stretch the FISH channels using the evaluated extrema
    #! This works with our data only
    if channels == 4:
        fish_647_to_analyze = contrast_stretch(
            data[FISH_647_CH],
            in_range=(min_intensity, max_intensity),
            mask=nuclei_labels_mask,
            ch_id=ch_dict[FISH_647_CH],
        )
    fish_568_to_analyze = contrast_stretch(
        data[FISH_568_CH],
        in_range=(min_intensity, max_intensity),
        mask=nuclei_labels_mask,
        ch_id=ch_dict[FISH_568_CH],
    )

    if visualize:
        #! This works with our data only
        if channels == 4:
            viewer.add_image(
                fish_647_to_analyze,
                name=ch_dict[FISH_647_CH] + "-analyzed-stretched",
                colormap="magenta",
                blending="additive",
                scale=spacing,
                interpolation="nearest",
                visible=False,
            )
        viewer.add_image(
            fish_568_to_analyze,
            name=ch_dict[FISH_568_CH] + "-analyzed-stretched",
            colormap="magenta",
            blending="additive",
            scale=spacing,
            interpolation="nearest",
            visible=False,
        )

    # Remove floor from FISH channels
    # Using a kernel about 3 times the size of the puncta
    if channels == 4:
        fish_647_to_analyze = remove_floor(
            fish_647_to_analyze,
            sigma=15 * np.array((1 / spacing_ratio, 1, 1)),
            filename_root=filename,
            ch_id=ch_dict[FISH_647_CH],
            mask=nuclei_labels_mask,
        )
    fish_568_to_analyze = remove_floor(
        fish_568_to_analyze,
        sigma=15 * np.array((1 / spacing_ratio, 1, 1)),
        filename_root=filename,
        ch_id=ch_dict[FISH_568_CH],
        mask=nuclei_labels_mask,
    )

    if visualize:
        #! This works with our data only
        if channels == 4:
            viewer.add_image(
                fish_647_to_analyze,
                name=ch_dict[FISH_647_CH] + "-analyzed-stretched-defloored",
                colormap="magenta",
                blending="additive",
                scale=spacing,
                interpolation="nearest",
                visible=False,
            )
        viewer.add_image(
            fish_568_to_analyze,
            name=ch_dict[FISH_568_CH] + "-analyzed-stretched-defloored",
            colormap="magenta",
            blending="additive",
            scale=spacing,
            interpolation="nearest",
            visible=False,
        )

    # # Apply median filter to denoise the FISH channels
    # #! This works with our data only
    # if channels == 4:
    #     fish_647_to_analyze = filter(
    #         fish_647_to_analyze,
    #         # footprint=ski_mor.ball(1),
    #         # footprint=ski_mor.ball(2)[1:4],
    #         # footprint=ski_mor.ball(3)[1::2],
    #         # footprint=ski_mor.ball(5)[2::3],
    #         footprint=ski_mor.ball(7)[3::4],
    #         filename_root=filename,
    #         ch_id=ch_dict[FISH_647_CH],
    #     )
    # fish_568_to_analyze = filter(
    #     fish_568_to_analyze,
    #     # footprint=ski_mor.ball(1),
    #     # footprint=ski_mor.ball(2)[1:4],
    #     # footprint=ski_mor.ball(3)[1::2],
    #     # footprint=ski_mor.ball(5)[2::3],
    #     footprint=ski_mor.ball(7)[3::4],
    #     filename_root=filename,
    #     ch_id=ch_dict[FISH_568_CH],
    # )

    # if visualize:
    # #! This works with our data only
    #     if channels == 4:
    #         viewer.add_image(
    #             fish_647_to_analyze,
    #             name=ch_dict[FISH_647_CH] + "-analyzed-stretched-defloored-median",
    #             colormap="magenta",
    #             blending="additive",
    #             scale=spacing,
    #             interpolation="nearest",
    #             visible=False,
    #         )
    #     viewer.add_image(
    #         fish_568_to_analyze,
    #         name=ch_dict[FISH_568_CH] + "-analyzed-stretched-defloored-median",
    #         colormap="magenta",
    #         blending="additive",
    #         scale=spacing,
    #         interpolation="nearest",
    #         visible=False,
    #     )

    # Find FISH signatures within channels
    #! This works with our data only
    if channels == 4:
        fish_647_puncta_df, props_647_df = get_fish_puncta(
            fish_647_to_analyze,
            nuclei_labels,
            nuclei_props_df,
            spacing_ratio,
            ch_id=ch_dict[FISH_647_CH],
            thresh_min=FISH_THRESHOLD_MIN,
            thresh_max=FISH_THRESHOLD_MAX,
            thresh_step=FISH_THRESHOLD_STEP,
            filename_root=filename,
        )
    fish_568_puncta_df, props_568_df = get_fish_puncta(
        fish_568_to_analyze,
        nuclei_labels,
        nuclei_props_df,
        spacing_ratio,
        ch_id=ch_dict[FISH_568_CH],
        thresh_min=FISH_THRESHOLD_MIN,
        thresh_max=FISH_THRESHOLD_MAX,
        thresh_step=FISH_THRESHOLD_STEP,
        filename_root=filename,
    )

    # Merge to nuclei props dataframe
    #! This works with our data only
    if channels == 4:
        nuclei_props_df = nuclei_props_df.merge(props_647_df, on="label", how="left")
    nuclei_props_df = nuclei_props_df.merge(props_568_df, on="label", how="left")

    # Fill missing counts with zeros and missing ids with empty lists
    filt = nuclei_props_df.filter(regex="cnt")
    nuclei_props_df[filt.columns] = filt.fillna(0)
    filt = nuclei_props_df.filter(regex="ids")
    nuclei_props_df[filt.columns] = filt.fillna(
        nuclei_props_df.notna().applymap(lambda x: x or [])
    )

    # Save the nuclei properties dataframe
    path = os_utils.build_path(
        filename,
        f"-{ch_dict[NUCLEI_CH]}-FISH-({FISH_THRESHOLD_MIN}-{FISH_THRESHOLD_MAX}-{FISH_THRESHOLD_STEP})-df",
    )
    nuclei_props_df.to_json(path + ".json")
    nuclei_props_df.to_csv(path + ".csv")

    if visualize:
        # Visualize puncta
        #! This works with our data only
        if channels == 4:
            pts_647 = viewer.add_points(
                fish_647_puncta_df[["Z", "Y", "X"]].to_numpy(),
                name=ch_dict[FISH_647_CH] + "-puncta",
                size=15,
                symbol="ring",
                opacity=1,
                scale=spacing,
                edge_color="blue",
                face_color="blue",
                blending="additive",
                out_of_slice_display=False,
                visible=False,
            )
            pts_647.features["label"] = fish_647_puncta_df["label"]
            pts_647.features["nucleus"] = fish_647_puncta_df["nucleus"]
            pts_647.features["thresholds"] = fish_647_puncta_df["thresholds"]

        pts_568 = viewer.add_points(
            fish_568_puncta_df[["Z", "Y", "X"]].to_numpy(),
            name=ch_dict[FISH_568_CH] + "-puncta",
            size=15,
            symbol="ring",
            opacity=1,
            scale=spacing,
            edge_color="red",
            face_color="red",
            blending="additive",
            out_of_slice_display=False,
            visible=False,
        )
        pts_568.features["label"] = fish_568_puncta_df["label"]
        pts_568.features["nucleus"] = fish_568_puncta_df["nucleus"]
        pts_568.features["thresholds"] = fish_568_puncta_df["thresholds"]

        # Create a widget to control threshold for puncta visualization
        #! This works with our data only
        if channels == 4:
            selected_thresholds = {
                ch_dict[FISH_647_CH] + "-puncta": FISH_THRESHOLD_MIN,
                ch_dict[FISH_568_CH] + "-puncta": FISH_THRESHOLD_MIN,
            }
        else:
            selected_thresholds = {
                ch_dict[FISH_568_CH] + "-puncta": FISH_THRESHOLD_MIN,
            }

        @magicgui(
            auto_call=True,
            threshold={
                "widget_type": "Slider",
                "min": FISH_THRESHOLD_MIN,
                "max": FISH_THRESHOLD_MAX,
                "step": FISH_THRESHOLD_STEP,
            },
        )
        def threshold_puncta(layer: napari.layers.Points, threshold: int) -> None:
            if "thresholds" in layer.features:
                threshold = 5 * (threshold // 5)
                selected_thresholds[layer.name] = threshold
                layer.shown = [threshold in f for f in layer.features["thresholds"]]

        #! This works with our data only
        if channels == 4:

            @magicgui(call_button="Display FISH")
            def display_fish() -> None:
                plt.close("all")
                # Plot 647 vs 538 for each nucleus
                top = max(
                    nuclei_props_df[
                        ch_dict[FISH_568_CH]
                        + f"_cnt_{selected_thresholds[ch_dict[FISH_568_CH] + '-puncta']:03}"
                    ].max(),
                    nuclei_props_df[
                        ch_dict[FISH_647_CH]
                        + f"_cnt_{selected_thresholds[ch_dict[FISH_647_CH] + '-puncta']:03}"
                    ].max(),
                )
                joint_plot = sbn.jointplot(
                    data=nuclei_props_df[nuclei_props_df["keep"]],
                    x=ch_dict[FISH_568_CH]
                    + f"_cnt_{selected_thresholds[ch_dict[FISH_568_CH] + '-puncta']:03}",
                    y=ch_dict[FISH_647_CH]
                    + f"_cnt_{selected_thresholds[ch_dict[FISH_647_CH] + '-puncta']:03}",
                    kind="hist",
                    ratio=3,
                    xlim=(0, top),
                    ylim=(0, top),
                    marginal_ticks=True,
                    cbar=True,
                    joint_kws={"binwidth": 1},
                    marginal_kws={"binwidth": 1},
                )

                # Move the colorbar to the right while keeping the plot square
                plt.subplots_adjust(left=0.1, right=0.8, top=0.8, bottom=0.1)
                pos_joint_ax = joint_plot.ax_joint.get_position()
                pos_marg_x_ax = joint_plot.ax_marg_x.get_position()
                joint_plot.ax_joint.set_position(
                    [
                        pos_joint_ax.x0,
                        pos_joint_ax.y0,
                        pos_marg_x_ax.width,
                        pos_joint_ax.height,
                    ]
                )
                joint_plot.fig.axes[-1].set_position(
                    [0.83, pos_joint_ax.y0, 0.07, pos_joint_ax.height]
                )
                # Enforce same limits and ticks on both axes
                joint_plot.ax_joint.set_yticks(joint_plot.ax_joint.get_xticks())
                joint_plot.ax_joint.set_xticks(joint_plot.ax_joint.get_yticks())
                joint_plot.ax_joint.set_xlim(0, joint_plot.ax_joint.get_xlim()[1])
                joint_plot.ax_joint.set_ylim(0, joint_plot.ax_joint.get_ylim()[1])
                # Add title and axes labels
                file = osp.split(filename)
                plt.suptitle(file[-1])
                plt.xlabel(
                    f"{ch_dict[FISH_568_CH]} count (thresh: {selected_thresholds[ch_dict[FISH_568_CH] + '-puncta']})"
                )
                plt.ylabel(
                    f"{ch_dict[FISH_647_CH]} count (thresh: {selected_thresholds[ch_dict[FISH_647_CH] + '-puncta']})"
                )
                plt.show()

        @magicgui(call_button="Export FISH")
        def export_fish() -> None:
            print("exporting")

        # Add the widgets
        viewer.window.add_dock_widget(threshold_puncta)
        viewer.window.add_dock_widget(export_fish)

        #! This works with our data only
        if channels == 4:
            viewer.window.add_dock_widget(display_fish)

        napari.run()


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
# #       - Blur segmented, find max, watershed with Vornoi boundaries:
# #         https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/Segmentation_3D.ipynb
# #         https://github.com/clEsperanto/pyclesperanto_prototype/blob/master/demo/segmentation/voronoi_otsu_labeling.ipynb)
# #       - Gaussian blur, then 3D Frangi to detect blobness then max of Frangi to detect nuclei
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
# #        - Ilastik (https://www.ilastik.org/) for cell segmentation
# #        - Imaris methods
# #        - 3D UNET, nnUNET
# #        - pyradiomics (https://pyradiomics.readthedocs.io/en/latest/index.html) for feature extraction
# #        - AroSpotFindingSuite (https://gitlab.com/evodevosys/AroSpotFindingSuite) for FISH
# #        - BlobFinder (software no longer availale but paper is: doi:10.1016/j.cmpb.2008.08.006)
# #        - FISH-quant (https://code.google.com/archive/p/fish-quant/)
# #        - ImageM (from https://www.nature.com/articles/nprot.2013.109).
# #          ImageM and other MATLAB codes developed in our lab are available
# #          upon request (requests can be addressed to
# #          S.I. (shalev.itzkovitz@weizmann.ac.il),
# #          J.P.J. (j.junker@hubrecht.eu) and
# #          A.v.O. (a.oudenaarden@hubrecht.eu))


from glob import glob

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        help="CZI file to analyze. If not specified, the user will be asked to select a file.",
        default=None,
    )
    parser.add_argument(
        "--visualize",
        help="Display analysis results using napari. (Default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--refresh",
        help="Don't use and refresh existing stored analysis. (Default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--channels",
        help="Specify the number of channels inside the CZI file. (Default: 4 - Range: 3 -> 4)",
        type=int,
        choices=range(3, 5),
        default=4,
    )
    parser.add_argument(
        "--metadata",
        help="Only retrieve and display the metadata of the CZI file. (Default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--fish_range",
        help="Range (min, max) to use for the initial FISH contrast stretching. If not specified, the values will be extracted from each channel.",
        default=None,
    )
    parser.add_argument(
        "--no_cyto",
        help="Specifies that the cytoplasm channel is not available. (Default: False)",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    if args.visualize:
        plt.ion()

    analyze_image(
        args.file,
        visualize=args.visualize,
        refresh=args.refresh,
        channels=args.channels,
        metadata=args.metadata,
        fish_range=args.fish_range,
        no_cyto=args.no_cyto,
    )

    # files = [
    #     g
    #     for p in [
    #         "/Users/avaccari/Library/CloudStorage/GoogleDrive-avaccari@middlebury.edu/Shared drives/MarkD/FISH analysis/2023-01 LPLC2 FISH calibration/*.czi"
    #     ]
    #     for g in glob(p)
    # ]
    # files.sort()
    # for f in files:
    #     try:
    #         analyze_image(
    #             f,
    #             visualize=False,
    #         )
    #     except:
    #         pass
