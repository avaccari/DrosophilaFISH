import numpy as np
import skimage.feature as ski_fea

import os_utils


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
    overwrite=False,
    out_dir=None,
):
    print("Detecting blobs' centers... ", end="", flush=True)
    if z_y_x_ratio is not None:
        min_sigma = min_sigma * np.array(z_y_x_ratio)
        max_sigma = max_sigma * np.array(z_y_x_ratio)

    blobs_ctrs = os_utils.store_to_npy(
        ski_fea.blob_log,
        filename_root=filename_root,
        ch_id=ch_id,
        suffix=f"blb-{tuple(np.round(min_sigma, decimals=2))}-{tuple(np.round(max_sigma, decimals=2))}",
        func_args={
            "image": data,
            "min_sigma": min_sigma,
            "max_sigma": max_sigma,
            "num_sigma": num_sigma,
            "threshold": threshold,
            "overlap": overlap,
            "exclude_border": True,
        },
        overwrite=overwrite,
        out_dir=out_dir,
    )

    print("done!")

    return blobs_ctrs
