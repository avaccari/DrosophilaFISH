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
    resolution=None,
    visualize=False,
    channels=4,
    metadata_only=False,
    no_fish=False,
    fish_contrast_range=None,
    fish_threshold_range=[2, 10.5, 0.5],
    cytoplasm_ch=3,
    nuclei_ch=0,
    regenerate_pre=False,
    regenerate_nuclei=False,
    regenerate_fish=False,
    nuclei_sigma_range=[15, 25, 3],
    nuclei_threshold=20,
    out_dir=None,
):
    # Ask user to choose a file
    print(f"\n{Fore.RED}{Style.BRIGHT}--- Starting new analysis ---{Style.RESET_ALL}")
    if filename is None:
        filename = filedialog.askopenfilename(
            filetypes=[("CZI files", "*.czi"), ("Numpy files", "*.npy")],
        )
        if filename == "":
            raise ValueError("A .czi or .npy file should be provided for analysis.")

    # Open image, and load and show metadata
    print(f"Loading file {Style.BRIGHT}{Fore.GREEN}{filename}{Style.RESET_ALL}")
    image = Image(
        filename,
        resolution=resolution,
        metadata_only=metadata_only,
        required_channels=channels,
        nuclei_ch=nuclei_ch,
        cytoplasm_ch=cytoplasm_ch,
    )

    # Stop if we only want the metadata
    if metadata_only:
        return

    print(
        f"{Style.BRIGHT}{Fore.BLUE}########## Analysis parameters: #########{Style.RESET_ALL}"
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
        f"{Style.BRIGHT}FISH detection threshold range (start, end, step_size):{Style.RESET_ALL} {(fish_threshold_range[0], fish_threshold_range[1], fish_threshold_range[2])}"
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

    # --- Development only: crop data ---
    dds = [np.floor(d // 4).astype("uint16") for d in image.data.shape]
    dde = [np.ceil(d - d // 4).astype("uint16") for d in image.data.shape]
    image.data = image.data[:, dds[1] : dde[1], dds[2] : dde[2], dds[3] : dde[3]]
    # -------------------------------------

    # Show original data
    if visualize:
        # Show pre-processed data
        viewer = napari.Viewer(title=osp.split(filename)[1], ndisplay=3)
        viewer.add_image(
            image.data,
            channel_axis=0,
            name=[
                n + "-orig" for (c, n) in image.ch_dict.items() if isinstance(c, int)
            ],
            colormap=image.ch_dict["colormaps"],
            blending="additive",
            scale=image.scaling,
            depiction="volume",
            interpolation="nearest",
            visible=False,
        )

    # Start pre-processing #######################################################

    # Contrast stretch nuclei and cytoplasm
    for ch in image.ch_dict["others"]:
        image.data[ch] = contrast_stretch(image.data[ch], ch_id=image.ch_dict[ch])

    # Contrast stretch the FISH channels
    if "fish" in image.ch_dict and not no_fish:
        min_intensity, max_intensity = np.inf, -np.inf
        for ch in image.ch_dict["fish"]:
            min_intensity = min(min_intensity, image.data[ch].min())
            max_intensity = max(max_intensity, image.data[ch].max())
        for ch in image.ch_dict["fish"]:
            image.data[ch] = contrast_stretch(
                image.data[ch],
                ch_id=image.ch_dict[ch],
                in_range=(min_intensity, max_intensity)
                if fish_contrast_range is None
                else (fish_contrast_range[0], fish_contrast_range[1]),
            )

    # If needed, convert to uint8
    if image.data.dtype != "uint8":
        for ch in range(image.channels_no):
            print(
                f"Converting {Fore.GREEN}{Style.BRIGHT}{image.ch_dict[ch]}{Style.RESET_ALL} from uint16 to uint8...",
                end="",
                flush=True,
            )
            input_min, input_median, input_max = eval_stats(image.data[ch])
            image.data[ch] = image.data[ch] // 256
            converted_min, converted_median, converted_max = eval_stats(image.data[ch])
            print(
                f" {Style.BRIGHT}[{input_min}, {input_median}, {input_max}] => [{converted_min}, {converted_median}, {converted_max}]{Style.RESET_ALL}... done!"
            )
        image.data = image.data.astype("uint8")

    # Remove floor from each channel
    for ch in range(image.channels_no):
        image.data[ch] = remove_floor(
            image.data[ch],
            sigma=100 * np.array((1 / image.scale_ratio, 1, 1)),
            filename_root=filename,
            ch_id=image.ch_dict[ch],
            overwrite=regenerate_pre,
            out_dir=out_dir,
        )

    # Contrast stretch nuclei and cytoplasm
    for ch in image.ch_dict["others"]:
        image.data[ch] = contrast_stretch(image.data[ch], ch_id=image.ch_dict[ch])

    # Contrast stretch the FISH channels
    if "fish" in image.ch_dict and not no_fish:
        min_intensity, max_intensity = np.inf, -np.inf
        for ch in image.ch_dict["fish"]:
            min_intensity = min(min_intensity, image.data[ch].min())
            max_intensity = max(max_intensity, image.data[ch].max())
        for ch in image.ch_dict["fish"]:
            image.data[ch] = contrast_stretch(
                image.data[ch],
                ch_id=image.ch_dict[ch],
                in_range=(min_intensity, max_intensity),
            )

    if visualize:
        # Show pre-processed data
        viewer.add_image(
            image.data,
            channel_axis=0,
            name=[n + "-pre" for (c, n) in image.ch_dict.items() if isinstance(c, int)],
            colormap=image.ch_dict["colormaps"],
            blending="additive",
            scale=image.scaling,
            depiction="volume",
            interpolation="nearest",
            visible=False,
        )

    # Apply median filter to denoise the nuclei channel
    print("Denoising nuclei's channel:")
    nuclei_den = filter(
        image.data[image.ch_dict["Nuclei"]],
        footprint=ski_mor.ball(7)[3::4],
        filename_root=filename,
        ch_id=image.ch_dict[image.ch_dict["Nuclei"]],
        overwrite=regenerate_pre,
        out_dir=out_dir,
    )
    if visualize:
        viewer.add_image(
            nuclei_den,
            name=image.ch_dict[image.ch_dict["Nuclei"]] + "-den",
            colormap="green",
            blending="additive",
            scale=image.scaling,
            interpolation="nearest",
            visible=False,
        )

    if "Cytoplasm" in image.ch_dict:
        # Apply median filter to denoise the cytoplasm channel
        print("Denoising cytoplasm's channel:")
        cytoplasm_den = filter(
            image.data[image.ch_dict["Cytoplasm"]],
            footprint=ski_mor.ball(7)[3::4],
            filename_root=filename,
            ch_id=image.ch_dict[image.ch_dict["Cytoplasm"]],
            overwrite=regenerate_pre,
            out_dir=out_dir,
        )
        if visualize:
            viewer.add_image(
                cytoplasm_den,
                name=image.ch_dict[image.ch_dict["Cytoplasm"]] + "-den",
                colormap="gray",
                blending="additive",
                scale=image.scaling,
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
    nuclei_centers = detect_blobs(
        nuclei_den.astype("float32"),
        min_sigma=nuclei_sigma_range[0],
        max_sigma=nuclei_sigma_range[1],
        num_sigma=nuclei_sigma_range[2],
        z_y_x_ratio=(1 / image.scale_ratio, 1, 1),
        threshold=nuclei_threshold,
        filename_root=filename,
        ch_id=image.ch_dict[image.ch_dict["Nuclei"]],
        overwrite=regenerate_nuclei,
        out_dir=out_dir,
    )

    # # Create dataframe with nuclei centers and assign IDs
    # nuclei_centers_df = pd.DataFrame(
    #     nuclei_centers,
    #     columns=['Z', 'Y', 'X', 'sigmaZ', 'sigmaY', 'sigmaX']
    #     )
    # nuclei_centers_df['ID'] = range(1, 1 + len(nuclei_centers))

    np.set_printoptions(precision=2)
    print(
        f"Detected {Style.BRIGHT}{len(nuclei_centers)}{Style.RESET_ALL} centers:\n{nuclei_centers}"
    )
    if visualize:
        nuclei_centers_viz = viewer.add_points(
            nuclei_centers[:, :3],
            name=image.ch_dict[image.ch_dict["Nuclei"]] + "-centers",
            size=nuclei_centers[:, 3:],
            symbol="disc",
            opacity=1,
            scale=image.scaling,
            edge_color="green",
            face_color="green",
            blending="additive",
            out_of_slice_display=True,
            visible=False,
        )
        nuclei_centers_viz.features["sigma"] = list(nuclei_centers[:, 3:])

    # If no centers were detected, stop the analysis
    if len(nuclei_centers) == 0:
        print("No nuclei's centers were detected. Stopping analysis.")
        return

    # Split the image volume in Voronoi cells based on the nuclei's center
    print("Identifying volumes associated with nuclei's centers:")
    nuclei_regions = evaluate_voronoi(
        np.ones_like(image.data[image.ch_dict["Nuclei"]], dtype="bool"),
        nuclei_centers[:, :3],
        spacing=(image.scale_ratio, 1, 1),
        filename_root=filename,
        ch_id="Volume",
        overwrite=regenerate_nuclei,
        out_dir=out_dir,
    )
    if visualize:
        nuclei_vor = viewer.add_labels(
            nuclei_regions,
            name=image.ch_dict[image.ch_dict["Nuclei"]] + "-vor",
            scale=image.scaling,
            blending="additive",
            visible=True,
        )
        nuclei_vor.contour = 2

    # Segment the nuclei
    nuclei = NucleiSegmentation(
        filename_root=filename,
        ch_id=image.ch_dict[image.ch_dict["Nuclei"]],
        overwrite=regenerate_nuclei,
        out_dir=out_dir,
    )
    nuclei_labels = nuclei.segment(
        labels=nuclei_regions,
        values=nuclei_den,
        centers=nuclei_centers[:, :3],
        write_to_tiff=True,
    )
    if visualize:
        nuclei_viz = viewer.add_labels(
            nuclei_labels,
            name=image.ch_dict[image.ch_dict["Nuclei"]] + "-labels",
            scale=image.scaling,
            blending="additive",
            visible=True,
        )
        nuclei_viz.contour = 2

    # Segment the cytoplasm
    # cytoplasm = CytoplasmSegmentation(regions=nuclei_regions, value=cytoplasm_den)
    # cytoplasm_labels = cytoplasm.segment()
    # cytoplasm_labels = segment_region(
    #     nuclei_regions,
    #     np.invert(cytoplasm_den),
    #     filename_root=filename,
    #     ch_id=image.ch_dict[image.ch_dict["Cytoplasm"]],
    # )
    # if visualize:
    #     cytoplasm_viz = viewer.add_labels(
    #         cytoplasm_labels,
    #         name=image.ch_dict[image.ch_dict["Cytoplasm"]] + "-labels",
    #         scale=image.scaling,
    #         blending="additive",
    #         visible=True,
    #     )
    #     cytoplasm_viz.contour = 2

    # Evaluate the overall nuclei mask
    nuclei_labels_mask = nuclei_labels > 0
    if visualize:
        viewer.add_image(
            nuclei_labels_mask,
            name=image.ch_dict[image.ch_dict["Nuclei"]] + "-msk",
            opacity=0.5,
            scale=image.scaling,
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
    # #     ch_id=image.ch_dict[image.ch_dict["Nuclei"]],
    # # )
    #
    # if visualize:
    #     nuclei_viz = viewer.add_labels(
    #         nuclei_labels,
    #         name=image.ch_dict[image.ch_dict["Nuclei"]] + "-labels-dilate",
    #         scale=image.scaling,
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

    # Save the nuclei properties dataframe
    path = os_utils.build_path(
        filename,
        f"-{image.ch_dict[image.ch_dict['Nuclei']]}-df",
        out_dir=out_dir,
    )
    nuclei_props_df.to_json(path + ".json")
    nuclei_props_df.to_csv(path + ".csv")

    # FISH detection ##############################################################
    if no_fish:
        print(f"{Fore.RED}{Style.BRIGHT}--- Analysis finished ---{Style.RESET_ALL}\n\n")
        if visualize:
            napari.run()
        return

    # Identify FISH puncta
    print("Identifying FISH puncta's centers within nuclei...")
    # To minimize difference between channels, we are pre-building the images
    # with just the region that we will analyze and then contrast stretch
    # globally. This should normalize the puncta intensity among channels as
    # well as within the channels. Contrast stretching individual nuclei
    # area would be equivalent to use non uniform detection thresholds.

    # Extract the subset of values from teh FISH channels within the identifies
    # nuclei bounding boxes and calculate extremes
    sub = image.data[image.ch_dict["fish"]][
        np.stack([nuclei_labels_mask] * len(image.ch_dict["fish"]))
    ]
    min_intensity = sub.min()
    max_intensity = sub.max()

    # Contrast stretch the FISH channels using the evaluated extrema
    fish_to_analyze = {}
    for ch in image.ch_dict["fish"]:
        fish_to_analyze[ch] = contrast_stretch(
            image.data[ch],
            ch_id=image.ch_dict[ch],
            in_range=(min_intensity, max_intensity),
            mask=nuclei_labels_mask,
        )
    if visualize:
        for ch in image.ch_dict["fish"]:
            viewer.add_image(
                fish_to_analyze[ch],
                name=image.ch_dict[ch] + "-analyzed-stretched",
                colormap="magenta",
                blending="additive",
                scale=image.scaling,
                interpolation="nearest",
                visible=False,
            )

    # Remove floor from FISH channels
    # Using a kernel about 3 times the size of the puncta
    for ch in image.ch_dict["fish"]:
        fish_to_analyze[ch] = remove_floor(
            fish_to_analyze[ch],
            sigma=15 * np.array((1 / image.scale_ratio, 1, 1)),
            filename_root=filename,
            ch_id=image.ch_dict[ch],
            mask=nuclei_labels_mask,
            overwrite=regenerate_fish,
            out_dir=out_dir,
        )
    if visualize:
        for ch in image.ch_dict["fish"]:
            viewer.add_image(
                fish_to_analyze[ch],
                name=image.ch_dict[ch] + "-analyzed-stretched-no-floor",
                colormap=image.ch_dict["colormaps"][ch],
                blending="additive",
                scale=image.scaling,
                interpolation="nearest",
                visible=True,
            )

    # Find FISH signatures within channels
    fish_puncta_df = {}
    props_df = {}
    for ch in image.ch_dict["fish"]:
        fish_puncta_df[ch], props_df[ch] = get_fish_puncta(
            fish_to_analyze[ch],
            nuclei_labels,
            nuclei_props_df,
            image.scale_ratio,
            ch_id=image.ch_dict[ch],
            thresh_min=fish_threshold_range[0],
            thresh_max=fish_threshold_range[1],
            thresh_step=fish_threshold_range[2],
            filename_root=filename,
            overwrite=regenerate_fish,
            out_dir=out_dir,
        )

    # ###########################################################
    # # Extract individual puncta
    # for ch in image.ch_dict["fish"]:
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
    #         chunk = image.data[ch][slc[0], slc[1], slc[2]]
    # #########################################################

    # Merge FISH to nuclei props dataframe
    for ch in image.ch_dict["fish"]:
        if not props_df[ch].empty:
            nuclei_props_df = nuclei_props_df.merge(
                props_df[ch], on="label", how="left"
            )

    # Fill missing counts with zeros and missing ids with empty lists
    filtered = nuclei_props_df.filter(regex="cnt")
    nuclei_props_df[filtered.columns] = filtered.fillna(0)
    filtered = nuclei_props_df.filter(regex="ids")
    nuclei_props_df[filtered.columns] = filtered.fillna(
        nuclei_props_df.notna().applymap(lambda x: x or [])
    )

    # Save the nuclei + FISH properties dataframe
    path = os_utils.build_path(
        filename,
        f"-{image.ch_dict[image.ch_dict['Nuclei']]}-FISH-df-({fish_threshold_range[0]}-{fish_threshold_range[1]}-{fish_threshold_range[2]})",
        out_dir=out_dir,
    )
    nuclei_props_df.to_json(path + ".json")
    nuclei_props_df.to_csv(path + ".csv")

    print(f"{Fore.RED}{Style.BRIGHT}--- Analysis finished ---{Style.RESET_ALL}\n\n")

    if visualize:
        # Visualize puncta
        for ch in image.ch_dict["fish"]:
            pts = viewer.add_points(
                fish_puncta_df[ch][["Z", "Y", "X"]].to_numpy(),
                name=image.ch_dict[ch] + "-puncta",
                size=10
                * fish_puncta_df[ch][["sigma_Z", "sigma_Y", "sigma_X"]].to_numpy(),
                symbol="ring",
                opacity=1,
                scale=image.scaling,
                edge_color=napari.utils.colormaps.bop_colors.bopd[
                    image.ch_dict["colormaps"][ch]
                ][1][-1],
                face_color=napari.utils.colormaps.bop_colors.bopd[
                    image.ch_dict["colormaps"][ch]
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
        for ch in image.ch_dict["fish"]:
            selected_thresholds[image.ch_dict[ch] + "-puncta"] = fish_threshold_range[0]

        @magicgui(
            auto_call=True,
            threshold={
                "widget_type": "FloatSlider",
                "min": fish_threshold_range[0],
                "max": fish_threshold_range[1],
                "step": fish_threshold_range[2],
            },
        )
        def threshold_puncta(layer: napari.layers.Points, threshold: float) -> None:
            if "thresholds" in layer.features:
                threshold = fish_threshold_range[2] * (
                    threshold // fish_threshold_range[2]
                )
                selected_thresholds[layer.name] = threshold
                layer.shown = [threshold in f for f in layer.features["thresholds"]]

        # Add the widgets
        viewer.window.add_dock_widget(threshold_puncta)

        napari.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        help="CZI or NPY file to analyze. If not specified, the user will be asked to select a file.",
        default=None,
    )
    parser.add_argument(
        "--resolution",
        help="Override the z y x resolution for the voxels. If not specified the values will be extracted from the image, if available. If not it will default to 1 1 1. (Pass as 3 space-separated values).",
        nargs=3,
        type=float,
        default=None,
    )
    parser.add_argument(
        "--channels",
        help="Specify the number of channels inside the image. If 1, it is assumed that only the nuclei channel is present. (Default: 4)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--nuclei_ch",
        help="Specifies the channel number where the nuclei are imaged. (Default: 0)",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--cytoplasm_ch",
        help="Specifies the channel number where the cytoplasm is imaged. Use `None` if the cytoplasm is not imaged. (Default: 3)",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--metadata_only",
        help="Only retrieve and display the metadata of the CZI file. (Default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--visualize",
        help="Display analysis results using napari. (Default: False)",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--nuclei_sigma_range",
        help="Range min max steps to use as LOG sigmas for the nuclei detection. (Default: 15 25 3. Pass as 3 space-separated values).",
        nargs=3,
        type=int,
        default=[10, 25, 3],
    )
    parser.add_argument(
        "--nuclei_threshold",
        help="Threshold to use in LOG for the nuclei detection. (Default: 20)",
        type=float,
        default=20,
    )
    parser.add_argument(
        "--fish_contrast_range",
        help="Range min max to use for the initial FISH raw data contrast stretching. If not specified, the values will be extracted from each channel. (Pass as 2 space-separated values).",
        nargs=2,
        type=int,
        default=None,
    )
    parser.add_argument(
        "--fish_threshold_range",
        help="Range min max steps to use for the thresholds used to detect FISH signatures. (Default: 2 10.5 0.5. Pass as 3 space-separated values).",
        nargs=3,
        type=float,
        default=[2, 10.5, 0.5],
    )
    parser.add_argument(
        "--no_fish",
        help="Don't perform the FISH detection. (Default: False)",
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
        resolution=args.resolution,
        visualize=args.visualize,
        channels=args.channels,
        metadata_only=args.metadata_only,
        fish_contrast_range=args.fish_contrast_range,
        fish_threshold_range=args.fish_threshold_range,
        cytoplasm_ch=args.cytoplasm_ch,
        nuclei_ch=args.nuclei_ch,
        no_fish=args.no_fish,
        regenerate_pre=args.regenerate_pre,
        regenerate_nuclei=args.regenerate_nuclei,
        regenerate_fish=args.regenerate_fish,
        nuclei_sigma_range=args.nuclei_sigma_range,
        nuclei_threshold=args.nuclei_threshold,
        out_dir=args.output_dir,
    )
