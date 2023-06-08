import os
import os.path as osp

import numpy as np
import pandas as pd
import scipy.spatial as sci_spa
from colorama import Fore, Style

import os_utils
from features import detect_blobs


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
    MIN_PUNCTA_SIGMA = 1.5
    MAX_PUNCTA_SIGMA = 3
    NUM_PUNCTA_SIGMA = 7

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
                sl[0].start - 4 * MAX_PUNCTA_SIGMA
                if sl[0].start >= 4 * MAX_PUNCTA_SIGMA
                else 0,
                sl[0].stop + 4 * MAX_PUNCTA_SIGMA
                if sl[0].stop <= fish_channel.shape[0] - 4 * MAX_PUNCTA_SIGMA
                else fish_channel.shape[0],
                None,
            )
            slice_1 = slice(
                sl[1].start - 4 * MAX_PUNCTA_SIGMA
                if sl[1].start >= 4 * MAX_PUNCTA_SIGMA
                else 0,
                sl[1].stop + 4 * MAX_PUNCTA_SIGMA
                if sl[1].stop <= fish_channel.shape[1] - 4 * MAX_PUNCTA_SIGMA
                else fish_channel.shape[1],
                None,
            )
            slice_2 = slice(
                sl[2].start - 4 * MAX_PUNCTA_SIGMA
                if sl[2].start >= 4 * MAX_PUNCTA_SIGMA
                else 0,
                sl[2].stop + 4 * MAX_PUNCTA_SIGMA
                if sl[2].stop <= fish_channel.shape[2] - 4 * MAX_PUNCTA_SIGMA
                else fish_channel.shape[2],
                None,
            )
            sl_buf = [slice_0, slice_1, slice_2]

            # Find 647 blobs locations within the bbox, mask, and shift coordinates
            # TODO: consider using a range of sigmas and then Frangi to preserve the most blobby
            ctrs = detect_blobs(
                fish_channel[sl_buf[0], sl_buf[1], sl_buf[2]].astype("float64"),
                min_sigma=MIN_PUNCTA_SIGMA,
                max_sigma=MAX_PUNCTA_SIGMA,
                num_sigma=NUM_PUNCTA_SIGMA,
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
                f"Detected {Fore.BLUE}{len(df):3d} puncta{Style.RESET_ALL} ({len(df) / n_row['area']:.4f} puncta/pixel)\033[F\033[A",
                end="",
            )

        # Assumption: if there are no detections at a given threshold, there
        # will be none at higher thresholds.
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
                #             f'{k + 1}, {r["X"]}, {r["Y"]}, {r["Z"]}, Marker{k + 1}, red\n'
                #         )
                props_df.to_json(file_props + ".json")
                props_df.to_csv(file_props + ".csv")
            except:
                print("WARNING: error saving the file.")

    return fish_puncta_df, props_df
