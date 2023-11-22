import matplotlib.pyplot as plt
import numpy as np
import skimage.measure as ski_mea
import skimage.morphology as ski_mor
from colorama import Fore, Style


import os_utils


class Cost:
    def __init__(self):
        pass

    def add(self, cost_dict):
        for k, v in cost_dict.items():
            if not hasattr(self, k):
                setattr(self, k, [])
            getattr(self, k).append(v)

    def get(self):
        return self.__dict__

    def show(self, ax=None):
        if ax is None:
            plt.ion()
            plt.figure()
            for k, v in self.get().items():
                plt.scatter(range(len(v)), v, s=3, label=k)

            plt.yscale("log")
            plt.legend()
            plt.show()
        else:
            for k, v in self.get().items():
                ax.scatter(range(len(v)), v, s=3, label=k)
            ax.set_yscale("log")
            ax.set_xticklabels([])
            ax.set_yticklabels([])


class NucleiSegmentation:
    def __init__(self, filename_root, ch_id, overwrite=False, out_dir=None):
        self.filename_root = filename_root
        self.ch_id = ch_id
        self.overwrite = overwrite
        self.out_dir = out_dir
        self.costs = {}

    def _find_rows_cols(self, rng, aspect_ratio=1.4):
        rows = 1
        cols = 1
        while rows * cols < rng:
            if cols / rows > aspect_ratio:
                rows += 1
            else:
                cols += 1
        return rows, cols

    def plot_costs(self, aspect_ratio=1.4):
        rows, cols = self._find_rows_cols(len(self.costs), aspect_ratio)
        plt.ion()
        fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True)
        for ax, (k, v) in zip(axs.flatten(), self.costs.items()):
            v.show(ax=ax)
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()
        plt.show()

    def _cost(self, t, temp_region, temp_values, temp_cytoplasm=None, normalize=False):
        # Assumes nuclei to be brighter than background
        # --- Nuclei ---
        # Threshold volume within the Voronoi cell
        msk = np.zeros_like(temp_region, dtype="bool")
        msk[temp_region] = temp_values[temp_region] >= t

        # Nucleus inside of volume
        in_idx = np.logical_and(msk, temp_region)
        in_variance = self._evaluate_variance_in_volume(
            in_idx, temp_values, normalize=normalize
        )

        # Nucleus outside of volume
        out_idx = np.logical_and(~msk, temp_region)
        out_variance = self._evaluate_variance_in_volume(
            out_idx, temp_values, normalize=normalize
        )

        # Cost if only nucleus is considered
        cost = out_variance + in_variance

        # --- Cytoplasm ---
        if temp_cytoplasm is not None:
            # # Threshold within the Voronoi cell
            # msk_cyto = np.zeros_like(temp_region, dtype="bool")
            # msk_cyto[temp_region] = temp_cytoplasm[temp_region] >= t

            # Cytoplasm outside of volume (expected inside the cytoplasm)
            in_cyto_variance = self._evaluate_variance_in_volume(
                out_idx, temp_cytoplasm, normalize=normalize
            )
            # Cytoplasm inside of volume (expected outside the cytoplasm)
            out_cyto_variance = self._evaluate_variance_in_volume(
                in_idx, temp_cytoplasm, normalize=normalize
            )

            # Cost if also cytospasm is considered
            cost = out_variance + out_cyto_variance
            # cost += out_cyto_variance

        surface_idx = np.logical_xor(
            ski_mor.binary_dilation(msk, footprint=ski_mor.ball(1)), msk
        )
        surface = surface_idx.sum()

        # # Minimize surface as well
        cost -= surface

        # volume = in_idx.sum()
        # volume_val = temp_values[in_idx].sum()

        table = ski_mea.regionprops(ski_mea.label(msk))
        objects = len(table)

        # cost += 0 if (objects == 1) else np.inf

        # Store costs in dictionary
        cost_dict = {
            "out_variance": out_variance,
            "in_variance": in_variance,
            "in_cyto_variance": in_cyto_variance if temp_cytoplasm is not None else 0,
            "out_cyto_variance": out_cyto_variance if temp_cytoplasm is not None else 0,
            "surface": surface,
            # "volume": volume,
            # "volume_val": volume_val,
            # "objects": objects,
            "total_cost": cost,
        }
        return (
            cost,
            objects,
            cost_dict,
        )

    def _evaluate_variance_in_volume(self, indexes, values, normalize=False):
        volume = indexes.sum()
        intensities = values[indexes]
        intensities_sum = intensities.sum()
        intensities_avg = 0 if volume == 0 else intensities_sum / volume
        intensities_var = np.square(intensities - intensities_avg)
        variance = intensities_var.sum()
        if normalize:
            variance = (variance / volume) if volume > 0 else 0

        return variance

    def _find_min_threshold(self, temp_region, temp_values, temp_cytoplasm=None):
        costs = Cost()
        min_cost = np.inf
        threshold_min = 0
        for threshold in range(1, 256):
            cost, objects, cost_dict = self._cost(
                threshold, temp_region, temp_values, temp_cytoplasm
            )
            costs.add(cost_dict)
            # Assumption: we are increasing the threshold so, if we have no
            # objects at a certain threshold, we will never have objects at
            # higher thresholds.
            if objects == 0:
                break
            if cost < min_cost:
                min_cost = cost
                threshold_min = threshold
            print(
                f"\rThreshold: {threshold:3d} => Cost: {cost:12.2f} Objects: {objects:3d} (Optimal: {threshold_min:3d} => {min_cost:12.2f})",
                end="",
                flush=True,
            )

        return threshold_min, costs

    def _find_region_limits(self, current_region):
        # Find extension of the region of interest within the whole volume
        z = np.any(current_region, axis=(1, 2))
        y = np.any(current_region, axis=(0, 2))
        x = np.any(current_region, axis=(0, 1))
        z_min, z_max = np.where(z)[0][[0, -1]]
        y_min, y_max = np.where(y)[0][[0, -1]]
        x_min, x_max = np.where(x)[0][[0, -1]]

        # Adjust z_max to include the last value excluded by slicing
        z_max += 1
        y_max += 1
        x_max += 1

        return (
            z_min,
            z_max,
            y_min,
            y_max,
            x_min,
            x_max,
        )

    def _get_centers_in_region(self, centers, z_min, z_max, y_min, y_max, x_min, x_max):
        return centers[
            np.logical_and.reduce(
                np.logical_and(
                    np.logical_and(
                        centers >= [z_min, y_min, x_min],
                        centers < [z_max, y_max, x_max],
                    ),
                    True,
                ),
                axis=1,
            ),
            :,
        ].astype("uint16") - [z_min, y_min, x_min]

    def _touch_edges(self, temp_mask_open):
        # Check that the region doesn't touch the border of the volume
        # TODO: modify to check if it touches the border of the image, not the volume
        return (
            temp_mask_open[0, :, :].any()
            or temp_mask_open[-1, :, :].any()
            or temp_mask_open[:, 0, :].any()
            or temp_mask_open[:, -1, :].any()
            or temp_mask_open[:, :, 0].any()
            or temp_mask_open[:, :, -1].any()
        )

    def _check_mask(
        self, temp_mask_open, centers, z_min, z_max, y_min, y_max, x_min, x_max
    ):
        # Get the centers inside the region of interest
        centers_in_region = self._get_centers_in_region(
            centers, z_min, z_max, y_min, y_max, x_min, x_max
        )

        # Check if there is at least one center inside the region of interest
        check = temp_mask_open[
            centers_in_region[:, 0], centers_in_region[:, 1], centers_in_region[:, 2]
        ]
        if check.any() == False:  # No center inside the region of interest
            return False, "center out of region"
        if check.sum() > 1:  # More than one center inside the region of interest
            return False, "multiple centers in region"
        # if self._touch_edges(temp_mask_open):  # Region touches the border of the image
        #     return False, "region touches border of image"
        return True, "good"

    def _segment(self, regions, values, centers, cytoplasm=None):
        labels = np.zeros_like(regions)
        # Region #0 is the background
        start = 1
        for lbl in range(start, regions.max() + start):
            print(f"Segment {self.ch_id}: {lbl:3d}/{regions.max():3d}")
            # Create a mask that isolates just the region of interest (ones)
            # everything else is zero.
            current_region = regions == lbl
            # Find the limits of the smallest volumes that include the region of interest
            (
                z_min,
                z_max,
                y_min,
                y_max,
                x_min,
                x_max,
            ) = self._find_region_limits(current_region)

            # Create temporary masks for the smallest volume that includes
            # the region of interest
            temp_region = current_region[z_min:z_max, y_min:y_max, x_min:x_max]
            temp_values = values[z_min:z_max, y_min:y_max, x_min:x_max]
            temp_cytoplasm = (
                None
                if cytoplasm is None
                else cytoplasm[z_min:z_max, y_min:y_max, x_min:x_max]
            )

            # Optimize the threshold to identify the nucleus within the region
            threshold_min, costs = self._find_min_threshold(
                temp_region, temp_values, temp_cytoplasm
            )
            self.costs[lbl] = costs

            # Create a mask that isolates the nucleus within the region of interest
            temp_mask = np.zeros_like(temp_region, dtype="bool")
            temp_mask[temp_region] = temp_values[temp_region] >= threshold_min

            # Open the mask. If this results in more than one component, keep the largest
            temp_mask_open = ski_mor.opening(temp_mask, footprint=ski_mor.ball(5)[2::3])
            temp_mask_open_labels = ski_mea.label(temp_mask_open)
            components = ski_mea.regionprops(temp_mask_open_labels)
            if len(components) > 1:
                area_max = np.max([c.area for c in components])
                for c in components:
                    if c.area < area_max:
                        temp_mask_open[temp_mask_open_labels == c.label] = 0

            # Verify that the mask can be added to the labels
            result, reason = self._check_mask(
                temp_mask_open, centers, z_min, z_max, y_min, y_max, x_min, x_max
            )
            if result:
                labels[
                    z_min:z_max, y_min:y_max, x_min:x_max
                ] += lbl * temp_mask_open.astype("uint16")
                print(f"{Fore.GREEN} ✓ ({reason}){Style.RESET_ALL}")
            else:
                print(f"{Fore.RED} ✕ ({reason}){Style.RESET_ALL}")

        return labels

    def segment(self, labels, values, centers, cytoplasm=None, write_to_tiff=False):
        print(f"Segmenting {self.ch_id}")
        labels = os_utils.store_to_npy(
            self._segment,
            filename_root=self.filename_root,
            ch_id=self.ch_id,
            suffix="labels",
            func_args={
                "regions": labels,
                "values": values,
                "centers": centers,
                "cytoplasm": cytoplasm,
            },
            overwrite=self.overwrite,
            out_dir=self.out_dir,
        )
        # If we are writing to tiff
        if write_to_tiff:
            os_utils.write_to_tif(
                labels,
                filename_root=self.filename_root,
                ch_id=self.ch_id,
                suffix="labels",
                out_dir=self.out_dir,
            )
        print("done!")
        return labels
