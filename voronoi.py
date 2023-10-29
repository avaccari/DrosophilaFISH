import numpy as np
import scipy.spatial as sci_spa
from colorama import Fore, Style

import os_utils


def _evaluate_voronoi(mask, markers, spacing=(1, 1, 1)):
    markers_tree = sci_spa.KDTree(markers * spacing)
    mask_idx = mask.nonzero()
    mask_idx_array = np.vstack(mask_idx).T
    closest_marker = markers_tree.query(mask_idx_array * spacing, workers=-1)
    labels = np.zeros_like(mask, dtype="uint16")
    # The +1 is to start the labels at 1 instead of 0
    labels[mask_idx[0], mask_idx[1], mask_idx[2]] = closest_marker[1] + 1

    return labels


def evaluate_voronoi(
    mask,
    markers,
    spacing=(1, 1, 1),
    filename_root=None,
    ch_id=None,
    overwrite=False,
    out_dir=None,
):
    # Find the equivalent to Voronoi regions in the provided mask based on the
    # provided markers
    # Coordinates are normalized to the physical size using 'spacing'
    print(
        f"Identifying regions within {Fore.GREEN}{Style.BRIGHT}{ch_id}{Style.RESET_ALL}'s mask with Vornoi... ",
        end="",
        flush=True,
    )

    labels = os_utils.store_to_npy(
        _evaluate_voronoi,
        filename_root=filename_root,
        ch_id=ch_id,
        suffix="voronoi",
        func_args={
            "mask": mask,
            "markers": markers,
            "spacing": spacing,
        },
        overwrite=overwrite,
        out_dir=out_dir,
    )

    print("done!")

    return labels
