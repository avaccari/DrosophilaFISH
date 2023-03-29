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
#! Implement the "overwrite" arguments in the analyze_image() function.


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
    MIN_SIGMA = 1.5
    MAX_SIGMA = 3
    NUM_SIGMA = 7

    fish_puncta_df = pd.DataFrame()
    props_df = pd.DataFrame()
    for threshold in np.arange(thresh_min, thresh_max, thresh_step):
        detections_at_thrs = {}
        for n_idx, n_row in nuclei_props_df[nuclei_props_df["keep"]].iterrows():
            lbl = n_row["label"]
            print(
                f"--- Ch: {ch_id} - Thrs: {thresh_min} => {threshold} => {thresh_max} - {n_idx + 1:3}/{len(nuclei_props_df):3} (lbl: {lbl:3}) ---"
            )
            sl = n_row["slice"]

            # Add a little buffer around based on MAX_SIGMA
            slice_0 = slice(
                sl[0].start - 4 * MAX_SIGMA if sl[0].start >= 4 * MAX_SIGMA else 0,
                sl[0].stop + 4 * MAX_SIGMA
                if sl[0].stop <= fish_channel.shape[0] - 4 * MAX_SIGMA
                else fish_channel.shape[0],
                None,
            )
            slice_1 = slice(
                sl[1].start - 4 * MAX_SIGMA if sl[1].start >= 4 * MAX_SIGMA else 0,
                sl[1].stop + 4 * MAX_SIGMA
                if sl[1].stop <= fish_channel.shape[1] - 4 * MAX_SIGMA
                else fish_channel.shape[1],
                None,
            )
            slice_2 = slice(
                sl[2].start - 4 * MAX_SIGMA if sl[2].start >= 4 * MAX_SIGMA else 0,
                sl[2].stop + 4 * MAX_SIGMA
                if sl[2].stop <= fish_channel.shape[2] - 4 * MAX_SIGMA
                else fish_channel.shape[2],
                None,
            )
            sl_buf = [slice_0, slice_1, slice_2]

            # Find 647 blobs locations within the bbox, mask, and shift coordinates
            # TODO: consider using a range of sigmas and then Frangi to preserve the most blobby
            ctrs = detect_blobs(
                fish_channel[sl_buf[0], sl_buf[1], sl_buf[2]].astype("float64"),
                min_sigma=MIN_SIGMA,
                max_sigma=MAX_SIGMA,
                num_sigma=NUM_SIGMA,
                z_y_x_ratio=(1, 1, 1),
                # z_y_x_ratio=(1 / spacing_ratio, 1, 1),
                threshold=threshold,
            ) + (sl_buf[0].start, sl_buf[1].start, sl_buf[2].start, 0, 0, 0)

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
                f"Detected {Fore.BLUE}{len(df)} puncta{Style.RESET_ALL} ({len(df) / n_row['area']:.4f} puncta/pixel)\033[F\033[A",
                end="",
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
    print("\033[B\033[B\033[E")  # Move cursor down three lines
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
            f"-{ch_id}-puncta-df-({thresh_min}-{thresh_max}-{thresh_step})",
        )
        file_props = os_utils.build_path(
            filename_root,
            f"-{ch_id}-puncta-props-df-({thresh_min}-{thresh_max}-{thresh_step})",
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
                # # Save puncta locations in a format compatible with imagej
                # with open(file_puncta + ".points", "w") as file_imagej:
                #     for k, r in fish_puncta_df.iterrows():
                #         file_imagej.write(
                #             f'"point{k}": [ {r["X"]}, {r["Y"]}, {r["Z"]} ]\n'
                #         )
                props_df.to_json(file_props + ".json")
                props_df.to_csv(file_props + ".csv")
            except:
                print("WARNING: error saving the file.")

    return fish_puncta_df, props_df


def analyze_image(
    filename=None,
    visualize=False,
    channels=4,
    metadata=False,
    raw_fish_range=None,
    no_cyto=False,
    overwrite_json=False,
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
    ch_dict = {}
    if channels == 4:
        if no_cyto:
            raise ValueError(
                f"Configuration of channels ({channels}) and no_cyto ({no_cyto}) not supported."
            )
        ch_dict["Nuclei"] = 0
        ch_dict["Cytoplasm"] = 3
        ch_dict["others"] = [0, 3]
        ch_dict["fish"] = [1, 2]
        ch_dict[0] = "Nuclei"
        ch_dict[1] = "FISH_647"
        ch_dict[2] = "FISH_568"
        ch_dict[3] = "Cytoplasm"
        ch_dict["colormaps"] = ["green", "bop blue", "bop orange", "gray"]
    elif channels == 3:
        if no_cyto:
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
    else:
        raise ValueError(f"Number of channels ({channels}) not allowed.")

    # Specify threshold for nuclei detection
    # TODO: maybe this could be linked to some features of the channel:
    # TODO: SNR, contrast, luminosity, etc.
    NUCLEI_THRESHOLD = 10

    # Specify thresholds for FISH detection
    FISH_THRESHOLD_MIN = 2
    FISH_THRESHOLD_MAX = 10.5
    FISH_THRESHOLD_STEP = 0.5

    # Gather and report image information
    pixel_sizes = image.scaling
    spacing = pixel_sizes
    # spacing_ratio = int(np.ceil(spacing[0] / spacing[1]))
    spacing_ratio = spacing[0] / spacing[1]
    contrast = [[np.min(data[ch]), np.max(data[ch])] for ch in range(data.shape[0])]
    print(
        f"{Style.BRIGHT}{Fore.BLUE}########## Original data info: ##########{Style.RESET_ALL}"
    )
    print(f"{Style.BRIGHT}Image shape (CH, Z, Y, X):{Style.RESET_ALL}")
    print(f"  {data.shape}")
    print(f"{Style.BRIGHT}Channels:{Style.RESET_ALL}")

    for (k, v), c in zip(
        image.channels_meta.items(),
        [v for k, v in ch_dict.items() if isinstance(k, int)],
    ):
        print(f"  {k}: {c:9} <=> {v}")
    print(f"{Style.BRIGHT}Pixel sizes (Z, Y, X):{Style.RESET_ALL}")
    print(f"  {spacing}")
    print(f"{Style.BRIGHT}Spacing ratio (Z / X or Y):")
    print(f"  {spacing_ratio}")
    print(f"{Style.BRIGHT}Data type:{Style.RESET_ALL}")
    for k, v in image.type_meta.items():
        print(f"  {k} => {v}")
    print(f"{Style.BRIGHT}Data ranges:{Style.RESET_ALL}")
    for (c, n), r in zip(
        [(k, v) for k, v in ch_dict.items() if isinstance(k, int)], contrast
    ):
        print(f"  {c}: {n:9} => {r}")

    print(
        f"{Style.BRIGHT}{Fore.BLUE}#########################################{Style.RESET_ALL}"
    )

    # Stop if we only want the metadata
    if metadata:
        return

    # Contrast stretch nuclei and cyto
    for ch in ch_dict["others"]:
        data[ch] = contrast_stretch(data[ch], ch_id=ch_dict[ch])

    # Contrast stretch the FISH channels
    min_intensity, max_intensity = np.inf, -np.inf
    for ch in ch_dict["fish"]:
        min_intensity = min(min_intensity, data[ch].min())
        max_intensity = max(max_intensity, data[ch].max())
    for ch in ch_dict["fish"]:
        data[ch] = contrast_stretch(
            data[ch],
            ch_id=ch_dict[ch],
            in_range=(min_intensity, max_intensity)
            if raw_fish_range is None
            else (raw_fish_range[0], raw_fish_range[1]),
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
            name=[n + "-orig" for (c, n) in ch_dict.items() if isinstance(c, int)],
            colormap=ch_dict["colormaps"],
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
    for ch in ch_dict["others"]:
        data[ch] = contrast_stretch(data[ch], ch_id=ch_dict[ch])

    # Contrast stretch the FISH channels
    min_intensity, max_intensity = np.inf, -np.inf
    for ch in ch_dict["fish"]:
        min_intensity = min(min_intensity, data[ch].min())
        max_intensity = max(max_intensity, data[ch].max())
    for ch in ch_dict["fish"]:
        data[ch] = contrast_stretch(
            data[ch],
            ch_id=ch_dict[ch],
            in_range=(min_intensity, max_intensity),
        )
    if visualize:
        # Show pre-processed data
        viewer.add_image(
            data,
            channel_axis=0,
            name=[n + "-pre" for (c, n) in ch_dict.items() if isinstance(c, int)],
            colormap=ch_dict["colormaps"],
            blending="additive",
            scale=spacing,
            depiction="volume",
            interpolation="nearest",
            visible=False,
        )

    # Apply median filter to denoise the nuclei channel
    print("Denoising nuclei's channel:")
    nuclei_den = filter(
        data[ch_dict["Nuclei"]],
        footprint=ski_mor.ball(7)[3::4],
        filename_root=filename,
        ch_id=ch_dict[ch_dict["Nuclei"]],
    )
    if visualize:
        viewer.add_image(
            nuclei_den,
            name=ch_dict[ch_dict["Nuclei"]] + "-den",
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
            name=ch_dict[ch_dict["Nuclei"]] + "-msk",
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

    # TODO: with the current threshold, a lot of centers are found. It might be
    #       worth changing it. You should keep in mind that the thresholds are
    #       affected by the contrast in the image. If we change how we stretch
    #       the image, it will change the required threshold.

    # Detect the centers of the nuclei
    print("Detecting nuclei's centers:")
    nuclei_ctrs = detect_blobs(
        nuclei_den.astype("float32"),
        min_sigma=15,
        max_sigma=25,
        num_sigma=3,
        z_y_x_ratio=(1 / spacing_ratio, 1, 1),
        threshold=NUCLEI_THRESHOLD,
        filename_root=filename,
        ch_id=ch_dict[ch_dict["Nuclei"]],
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
    # TODO: Add the sigmas in the properties of the visualization
    if visualize:
        nuclei_ctrs_viz = viewer.add_points(
            nuclei_ctrs[:, :3],
            name=ch_dict[ch_dict["Nuclei"]] + "-ctrs",
            size=nuclei_ctrs[:, 3:],
            symbol="disc",
            opacity=1,
            scale=spacing,
            edge_color="green",
            face_color="green",
            blending="additive",
            out_of_slice_display=True,
            visible=False,
        )
        nuclei_ctrs_viz.features["sigma"] = list(nuclei_ctrs[:, 3:])

    # Find the Voronoi regions in the nuclei's mask using detected nuclei's
    # centers as markers
    nuclei_labels_voronoi = evaluate_voronoi(
        nuclei_mask,
        nuclei_ctrs[:, :3],
        spacing=(spacing_ratio, 1, 1),
        filename_root=filename,
        ch_id=ch_dict[ch_dict["Nuclei"]],
    )

    if visualize:
        viewer.add_labels(
            nuclei_labels_voronoi,
            name=ch_dict[ch_dict["Nuclei"]] + "-vor-lbls",
            scale=spacing,
            blending="additive",
            visible=False,
        )

    # Use watershed to identify regions connected to the nuclei centers
    nuclei_labels_watershed = evaluate_watershed(
        nuclei_mask,
        nuclei_ctrs[:, :3],
        filename_root=filename,
        ch_id=ch_dict[ch_dict["Nuclei"]],
    )
    if visualize:
        viewer.add_labels(
            nuclei_labels_watershed,
            name=ch_dict[ch_dict["Nuclei"]] + "-ws-lbls",
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
            name=ch_dict[ch_dict["Nuclei"]] + "-lbls-no-cyto",
            scale=spacing,
            blending="additive",
            visible=False,
        )
    print("done!")

    # If we have a cytoplasm channel
    if "Cytoplasm" in ch_dict:
        # Apply median filter to denoise the cytoplasm channel
        print("Denoising cytoplasm's channel:")
        cyto_den = filter(
            data[ch_dict["Cytoplasm"]],
            footprint=ski_mor.ball(7)[3::4],
            filename_root=filename,
            ch_id=ch_dict[ch_dict["Cytoplasm"]],
        )
        if visualize:
            viewer.add_image(
                cyto_den,
                name=ch_dict[ch_dict["Cytoplasm"]] + "-den",
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
            ch_id=ch_dict[ch_dict["Cytoplasm"]],
        )
        if visualize:
            viewer.add_image(
                cyto_closed,
                name=ch_dict[ch_dict["Cytoplasm"]] + "-closed",
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
                name=ch_dict[ch_dict["Cytoplasm"]] + "-msk",
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
                name=ch_dict[ch_dict["Cytoplasm"]] + "-msk-inv",
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
            ch_id=ch_dict[ch_dict["Cytoplasm"]],
        )
        if visualize:
            viewer.add_labels(
                cyto_inv_labels_voronoi,
                name=ch_dict[ch_dict["Cytoplasm"]] + "-inv-vor-lbls",
                scale=spacing,
                blending="additive",
                visible=False,
            )

        # Use watershed to identify regions connected to the nuclei centers
        cyto_inv_labels_watershed = evaluate_watershed(
            cyto_mask_inv,
            nuclei_ctrs[:, :3],
            filename_root=filename,
            ch_id=ch_dict[ch_dict["Cytoplasm"]],
        )
        if visualize:
            viewer.add_labels(
                cyto_inv_labels_watershed,
                name=ch_dict[ch_dict["Cytoplasm"]] + "-inv-ws-lbls",
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
                name=ch_dict[ch_dict["Nuclei"]] + "-lbls-with-cyto",
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
        ch_id=ch_dict[ch_dict["Nuclei"]],
    )
    if visualize:
        nuclei_viz = viewer.add_labels(
            nuclei_labels,
            name=ch_dict[ch_dict["Nuclei"]] + "-lbls-with-cyto-dilate",
            scale=spacing,
            blending="additive",
            visible=True,
        )
        nuclei_viz.contour = 2
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
    sub = data[ch_dict["fish"]][np.stack([nuclei_labels_mask] * len(ch_dict["fish"]))]
    min_intensity = sub.min()
    max_intensity = sub.max()

    # Contrast stretch the FISH channels using the evaluated extrema
    fish_to_analyze = {}
    for ch in ch_dict["fish"]:
        fish_to_analyze[ch] = contrast_stretch(
            data[ch],
            ch_id=ch_dict[ch],
            in_range=(min_intensity, max_intensity),
            mask=nuclei_labels_mask,
        )
    if visualize:
        for ch in ch_dict["fish"]:
            viewer.add_image(
                fish_to_analyze[ch],
                name=ch_dict[ch] + "-analyzed-stretched",
                colormap="magenta",
                blending="additive",
                scale=spacing,
                interpolation="nearest",
                visible=False,
            )

    # Remove floor from FISH channels
    # Using a kernel about 3 times the size of the puncta
    for ch in ch_dict["fish"]:
        fish_to_analyze[ch] = remove_floor(
            fish_to_analyze[ch],
            sigma=15 * np.array((1 / spacing_ratio, 1, 1)),
            filename_root=filename,
            ch_id=ch_dict[ch],
            mask=nuclei_labels_mask,
        )
    if visualize:
        for ch in ch_dict["fish"]:
            viewer.add_image(
                fish_to_analyze[ch],
                name=ch_dict[ch] + "-analyzed-stretched-defloored",
                colormap=ch_dict["colormaps"][ch],
                blending="additive",
                scale=spacing,
                interpolation="nearest",
                visible=True,
            )

    # Find FISH signatures within channels
    fish_puncta_df = {}
    props_df = {}
    for ch in ch_dict["fish"]:
        fish_puncta_df[ch], props_df[ch] = get_fish_puncta(
            fish_to_analyze[ch],
            nuclei_labels,
            nuclei_props_df,
            spacing_ratio,
            ch_id=ch_dict[ch],
            thresh_min=FISH_THRESHOLD_MIN,
            thresh_max=FISH_THRESHOLD_MAX,
            thresh_step=FISH_THRESHOLD_STEP,
            filename_root=filename,
        )

    # Merge to nuclei props dataframe
    for ch in ch_dict["fish"]:
        nuclei_props_df = nuclei_props_df.merge(props_df[ch], on="label", how="left")

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
        f"-{ch_dict[ch_dict['Nuclei']]}-df-({FISH_THRESHOLD_MIN}-{FISH_THRESHOLD_MAX}-{FISH_THRESHOLD_STEP})",
    )
    nuclei_props_df.to_json(path + ".json")
    nuclei_props_df.to_csv(path + ".csv")

    print(f"{Fore.RED}{Style.BRIGHT}--- Analysis finished ---{Style.RESET_ALL}\n\n")

    if visualize:
        # Visualize puncta
        for ch in ch_dict["fish"]:
            pts = viewer.add_points(
                fish_puncta_df[ch][["Z", "Y", "X"]].to_numpy(),
                name=ch_dict[ch] + "-puncta",
                size=10
                * fish_puncta_df[ch][["sigma_Z", "sigma_Y", "sigma_X"]].to_numpy(),
                symbol="ring",
                opacity=1,
                scale=spacing,
                edge_color=napari.utils.colormaps.bop_colors.bopd[
                    ch_dict["colormaps"][ch]
                ][1][-1],
                face_color=napari.utils.colormaps.bop_colors.bopd[
                    ch_dict["colormaps"][ch]
                ][1][-1],
                blending="additive",
                out_of_slice_display=False,
                visible=True,
            )
            pts.features["label"] = fish_puncta_df[ch]["label"]
            pts.features["sigmas"] = list(
                fish_puncta_df[ch][["sigma_Z", "sigma_Y", "sigma_X"]].to_numpy()
            )
            pts.features["nucleus"] = fish_puncta_df[ch]["nucleus"]
            pts.features["thresholds"] = fish_puncta_df[ch]["thresholds"]

        # Create a widget to control threshold for puncta visualization
        selected_thresholds = {}
        for ch in ch_dict["fish"]:
            selected_thresholds[ch_dict[ch] + "-puncta"] = FISH_THRESHOLD_MIN

        @magicgui(
            auto_call=True,
            threshold={
                "widget_type": "FloatSlider",
                "min": FISH_THRESHOLD_MIN,
                "max": FISH_THRESHOLD_MAX,
                "step": FISH_THRESHOLD_STEP,
            },
        )
        def threshold_puncta(layer: napari.layers.Points, threshold: float) -> None:
            if "thresholds" in layer.features:
                threshold = FISH_THRESHOLD_STEP * (threshold // FISH_THRESHOLD_STEP)
                selected_thresholds[layer.name] = threshold
                layer.shown = [threshold in f for f in layer.features["thresholds"]]

        # Add the widgets
        viewer.window.add_dock_widget(threshold_puncta)

        napari.run()


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
        "--raw_fish_range",
        help="Range (min, max) to use for the initial FISH raw data contrast stretching. If not specified, the values will be extracted from each channel.",
        default=None,
    )
    parser.add_argument(
        "--no_cyto",
        help="Specifies that the cytoplasm channel is not available. (Default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--overwrite_json",
        help="Ignore stored JSON files and overwrite them. (Default: False)",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    if args.visualize:
        plt.ion()

    analyze_image(
        args.file,
        visualize=args.visualize,
        channels=args.channels,
        metadata=args.metadata,
        raw_fish_range=args.raw_fish_range,
        no_cyto=args.no_cyto,
        overwrite_json=args.overwrite_json,
    )
