import argparse
import os.path as osp
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.measure as ski_mea
import skimage.morphology as ski_mor
from colorama import Fore, Style

import os_utils
from features import detect_blobs
from fish import get_fish_puncta
from image import Image
from preprocess import contrast_stretch, eval_stats, filter, remove_floor
from segment import NucleiSegmentation
from voronoi import evaluate_voronoi

#! IMPORTANT FOR CONSISTENCY: add a flag so that if there is an exception with a
#! file, all the following steps are re-evaluated instead of using stored values
#! Implement the "overwrite" arguments in the analyze_image() function.


def analyze_image(
    filename=None,
    visualize=False,
    channels=4,
    metadata=False,
    fish_contrast_range=None,
    no_cyto=False,
    nuclei_ch1=False,
    regenerate_pre=False,
    regenerate_nuclei=False,
    regenerate_fish=False,
    nuclei_sigma_range=(15, 25, 3),
    nuclei_threshold=20,
    out_dir=None,
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
    if not metadata and channels != image.channels_no:
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
        if nuclei_ch1:
            ch_dict["Nuclei"] = 1
            ch_dict["Cytoplasm"] = 3
            ch_dict["others"] = [1, 3]
            ch_dict["fish"] = [0, 2]
            ch_dict[0] = "FISH_647"
            ch_dict[1] = "Nuclei"
            ch_dict[2] = "FISH_568"
            ch_dict[3] = "Cytoplasm"
            ch_dict["colormaps"] = ["bop blue", "green", "bop orange", "gray"]
        else:
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
        if nuclei_ch1:
            raise ValueError(
                f"Configuration of channels ({channels}) and nuclei_ch1 ({nuclei_ch1}) not supported."
            )
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
    print(f"{Style.BRIGHT}Spacing ratio (Z / X or Y):{Style.RESET_ALL}")
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
    print(
        f"{Style.BRIGHT}{Fore.BLUE}\n########## Analysis parameters: #########{Style.RESET_ALL}"
    )
    print(
        f"{Style.BRIGHT}Nuclei detection sigma range (start, end, steps):{Style.RESET_ALL} {nuclei_sigma_range}"
    )
    print(
        f"{Style.BRIGHT}Nuclei detection threshold:{Style.RESET_ALL} {nuclei_threshold}"
    )
    print(
        f"{Style.BRIGHT}FISH contrast stretching range (None => full range):{Style.RESET_ALL} {fish_contrast_range}"
    )
    print(
        f"{Style.BRIGHT}FISH detection threshold range (start, end, step_size):{Style.RESET_ALL} {(FISH_THRESHOLD_MIN, FISH_THRESHOLD_MAX, FISH_THRESHOLD_STEP)}"
    )
    print(
        f"{Style.BRIGHT}Regenerate pre-processing data:{Style.RESET_ALL} {regenerate_pre}"
    )
    print(
        f"{Style.BRIGHT}Regenerate nuclei detection data:{Style.RESET_ALL} {regenerate_nuclei}"
    )
    print(
        f"{Style.BRIGHT}Regenerate FISH detection data:{Style.RESET_ALL} {regenerate_fish}"
    )
    print(f"{Style.BRIGHT}Output folder:{Style.RESET_ALL}")
    print(f"  {os_utils.build_path(filename, suffix=None, out_dir=out_dir)}")

    print(
        f"{Style.BRIGHT}{Fore.BLUE}#########################################{Style.RESET_ALL}"
    )

    # Stop if we only want the metadata
    if metadata:
        return

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

    # Start pre-processing #######################################################

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
            if fish_contrast_range is None
            else (fish_contrast_range[0], fish_contrast_range[1]),
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

    # Remove floor from each channel
    for ch in range(data.shape[0]):
        data[ch] = remove_floor(
            data[ch],
            sigma=100 * np.array((1 / spacing_ratio, 1, 1)),
            filename_root=filename,
            ch_id=ch_dict[ch],
            overwrite=regenerate_pre,
            out_dir=out_dir,
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
        overwrite=regenerate_pre,
        out_dir=out_dir,
    )
    if visualize:
        viewer.add_image(
            nuclei_den,
            name=ch_dict[ch_dict["Nuclei"]] + "-den",
            colormap="green",
            blending="additive",
            scale=spacing,
            interpolation="nearest",
            visible=False,
        )

    if not no_cyto:
        # Apply median filter to denoise the cytoplasm channel
        print("Denoising cytoplasm's channel:")
        cyto_den = filter(
            data[ch_dict["Cytoplasm"]],
            footprint=ski_mor.ball(7)[3::4],
            filename_root=filename,
            ch_id=ch_dict[ch_dict["Cytoplasm"]],
            overwrite=regenerate_pre,
            out_dir=out_dir,
        )
        if visualize:
            viewer.add_image(
                cyto_den,
                name=ch_dict[ch_dict["Cytoplasm"]] + "-den",
                colormap="gray",
                blending="additive",
                scale=spacing,
                interpolation="nearest",
                visible=False,
            )

    # Nuclei detection ###########################################################

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
    # TODO: maybe sigma range and threshold could be calculated from some
    # TODO: features of the nuclei channel:
    # TODO: SNR, contrast, luminosity, etc.
    print("Detecting nuclei's centers:")
    nuclei_ctrs = detect_blobs(
        nuclei_den.astype("float32"),
        min_sigma=nuclei_sigma_range[0],
        max_sigma=nuclei_sigma_range[1],
        num_sigma=nuclei_sigma_range[2],
        z_y_x_ratio=(1 / spacing_ratio, 1, 1),
        threshold=nuclei_threshold,
        filename_root=filename,
        ch_id=ch_dict[ch_dict["Nuclei"]],
        overwrite=regenerate_nuclei,
        out_dir=out_dir,
    )

    # # Create dataframe with nuclei centers and assign IDs
    # nuclei_ctrs_df = pd.DataFrame(
    #     nuclei_ctrs,
    #     columns=['Z', 'Y', 'X', 'sigmaZ', 'sigmaY', 'sigmaX']
    #     )
    # nuclei_ctrs_df['ID'] = range(1, 1 + len(nuclei_ctrs))

    np.set_printoptions(precision=2)
    print(
        f"Detected {Style.BRIGHT}{len(nuclei_ctrs)}{Style.RESET_ALL} centers:\n{nuclei_ctrs}"
    )
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

    # If no centers were detected, stop the analysis
    if len(nuclei_ctrs) == 0:
        print("No nuclei's centers were detected. Stopping analysis.")
        return

    # Split the image volume in Voronoi cells based on the nuclei's center
    print("Identifying volumes associated with nuclei's centers:")
    nuclei_regions = evaluate_voronoi(
        np.ones_like(data[ch_dict["Nuclei"]], dtype="bool"),
        nuclei_ctrs[:, :3],
        spacing=(spacing_ratio, 1, 1),
        filename_root=filename,
        ch_id="Volume",
        overwrite=regenerate_nuclei,
        out_dir=out_dir,
    )
    if visualize:
        nuclei_vor = viewer.add_labels(
            nuclei_regions,
            name=ch_dict[ch_dict["Nuclei"]] + "-vor",
            scale=spacing,
            blending="additive",
            visible=True,
        )
        nuclei_vor.contour = 2

    # Segment the nuclei
    nuclei = NucleiSegmentation(
        filename_root=filename,
        ch_id=ch_dict[ch_dict["Nuclei"]],
        overwrite=regenerate_nuclei,
        out_dir=out_dir,
    )
    nuclei_labels = nuclei.segment(
        labels=nuclei_regions,
        values=nuclei_den,
        centers=nuclei_ctrs[:, :3],
        write_to_tiff=True,
    )
    if visualize:
        nuclei_viz = viewer.add_labels(
            nuclei_labels,
            name=ch_dict[ch_dict["Nuclei"]] + "-lbls",
            scale=spacing,
            blending="additive",
            visible=True,
        )
        nuclei_viz.contour = 2

    # Segment the cytoplasm
    # cytoplasm = CytoplasmSegmentation(regions=nuclei_regions, value=cyto_den)
    # cyto_labels = cytoplasm.segment()
    # cyto_labels = segment_region(
    #     nuclei_regions,
    #     np.invert(cyto_den),
    #     filename_root=filename,
    #     ch_id=ch_dict[ch_dict["Cytoplasm"]],
    # )
    # if visualize:
    #     cyto_viz = viewer.add_labels(
    #         cyto_labels,
    #         name=ch_dict[ch_dict["Cytoplasm"]] + "-lbls",
    #         scale=spacing,
    #         blending="additive",
    #         visible=True,
    #     )
    #     cyto_viz.contour = 2

    # Evaluate the overall nuclei mask
    nuclei_labels_mask = nuclei_labels > 0
    if visualize:
        viewer.add_image(
            nuclei_labels_mask,
            name=ch_dict[ch_dict["Nuclei"]] + "-msk",
            opacity=0.5,
            scale=spacing,
            colormap="blue",
            blending="additive",
            interpolation="nearest",
            visible=False,
        )

    ################################################################################
    # Hack to be improved: Dilate preserved nuclei labels to identify nearby puncta
    # print("Dilate preserved nuclei to include part of the surrounding cytoplasm:")
    # # nuclei_labels = filter(
    # #     nuclei_labels,
    # #     type="maximum",
    # #     footprint=ski_mor.ball(5)[2::3],
    # #     # footprint=ski_mor.ball(7)[3::4],
    # #     # footprint=ski_mor.ball(9)[1::4],
    # #     filename_root=filename,
    # #     ch_id=ch_dict[ch_dict["Nuclei"]],
    # # )
    #
    # if visualize:
    #     nuclei_viz = viewer.add_labels(
    #         nuclei_labels,
    #         name=ch_dict[ch_dict["Nuclei"]] + "-lbls-dilate",
    #         scale=spacing,
    #         blending="additive",
    #         visible=True,
    #     )
    #     nuclei_viz.contour = 2
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

    # FISH detection ##############################################################

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
            overwrite=regenerate_fish,
            out_dir=out_dir,
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
            overwrite=regenerate_fish,
            out_dir=out_dir,
        )

    # ###########################################################
    # # Extract individual puncta
    # for ch in ch_dict["fish"]:
    #     for _, row in fish_puncta_df[ch].iterrows():
    #         coo = row[["Z", "Y", "X"]].astype("uint16")
    #         sig = (
    #             2 * np.sqrt(3) * row[["sigma_Z", "sigma_Y", "sigma_X"]].to_numpy() + 0.5
    #         ).astype("uint16")
    #         slc = [slice(c - s, c + s + 1) for c, s in zip(coo, sig)]
    #         grd = np.mgrid[slc[0], slc[1], slc[2]]
    #         dist = np.sqrt(
    #             np.square(grd[0] - coo[0])
    #             + np.square(grd[1] - coo[1])
    #             + np.square(grd[2] - coo[2])
    #         )
    #         chunk = data[ch][slc[0], slc[1], slc[2]]
    # #########################################################

    # Merge to nuclei props dataframe
    for ch in ch_dict["fish"]:
        if not props_df[ch].empty:
            nuclei_props_df = nuclei_props_df.merge(
                props_df[ch], on="label", how="left"
            )

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
        out_dir=out_dir,
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
        "--nuclei_sigma_range",
        help="Range (min, max, steps) to use as LOG sigmas for the nuclei detection. (Default: 15, 25, 3))",
        default=(15, 25, 3),
    )
    parser.add_argument(
        "--nuclei_threshold",
        help="Threshold to use in LOG for the nuclei detection. (Default: 20)",
        type=float,
        default=20,
    )
    parser.add_argument(
        "--fish_contrast_range",
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
        "--nuclei_ch1",
        help="Specifies that the nuclei are in channel 1 instead of channel 0. (Default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--regenerate_pre",
        help="Regenerate stored file associated with pre-processing. (Default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--regenerate_nuclei",
        help="Regenerate stored file associated with nuclei detection. (Default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--regenerate_fish",
        help="Regenerate stored file associated with FISH detection. (Default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--output_dir",
        help="Directory where to look for and store the folder containing results and auxiliary files. (Default: the same directory as the input file)",
    )

    args = parser.parse_args()

    if args.visualize:
        import napari
        from magicgui import magicgui

        plt.ion()

    analyze_image(
        args.file,
        visualize=args.visualize,
        channels=args.channels,
        metadata=args.metadata,
        fish_contrast_range=args.fish_contrast_range,
        no_cyto=args.no_cyto,
        nuclei_ch1=args.nuclei_ch1,
        regenerate_pre=args.regenerate_pre,
        regenerate_nuclei=args.regenerate_nuclei,
        regenerate_fish=args.regenerate_fish,
        nuclei_sigma_range=args.nuclei_sigma_range,
        nuclei_threshold=args.nuclei_threshold,
        out_dir=args.output_dir,
    )
