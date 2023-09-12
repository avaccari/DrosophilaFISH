import numpy as np
import skimage.feature as ski_fea

import os_utils


# Note: The radius of each blob is approximately sqrt(2) * sigma for 2D and
# sqrt(3) * sigma for 3D.
# Ex: LPLC2 nuclei are about 6.5um in diameter. If the resolution of the image
#     is 0.092um/px, then the nuclei are about 70px in diameter. If we want to
#     detect blobs of 6.5um in diameter, then we need to use a sigma of
#     70px/2/sqrt(3) = 20.2px. The search range should include this value.
#     For Giant Fiber nuclei, the largest diameter can be around 12um, so the
#     sigma should be 12um/0.092um/2/sqrt(3) = 34.6px (for images with the same
#     resolution). The main problem with LPLC2 is that the nuclei can be very
#     non-spherical.
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
