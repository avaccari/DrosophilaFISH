from abc import ABC, abstractmethod

import numpy as np
import skimage.measure as ski_mea
import skimage.morphology as ski_mor

import os_utils

import matplotlib.pyplot as plt


class Segmentation(ABC):
    def __init__(self, filename_root, ch_id):
        self.filename_root = filename_root
        self.ch_id = ch_id

    @abstractmethod
    def _cost(self, threshold, temp_region, temp_values):
        pass

    def _evaluate_variance_in_volume(self, indexes, values):
        volume = indexes.sum()
        intensities = values[indexes]
        intensities_sum = intensities.sum()
        intensities_avg = 0 if volume == 0 else intensities_sum / volume
        intensities_stdev = np.square(intensities - intensities_avg)

        return intensities_stdev.sum()

    def _find_min_threshold(self, start, temp_region, temp_values):
        costs = []  # Debug
        min_cost = np.inf
        threshold_min = 0
        for threshold in range(start, 256):
            cost, objects = self._cost(threshold, temp_region, temp_values)
            # Assumption: we are increasing the threshold so, if we have no
            # objects at a certain threshold, we will never have objects at
            # higher thresholds.
            if objects == 0:
                break
            if cost < min_cost:
                min_cost = cost
                threshold_min = threshold
            print(
                f"Threshold: {threshold:3d} => Cost: {cost:12.2f} (Optimal: {threshold_min:3d} => {min_cost:12.2f})\r",
                end="",
                flush=True,
            )
            costs.append(cost)
        plt.scatter(range(start, start + len(costs)), costs, s=1)  # Debug
        return threshold_min

    def _temp_masks(self, values, current_region):
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

        # Create temporary masks for the smallest volume that includes the
        # 'region of interest
        temp_region = current_region[z_min:z_max, y_min:y_max, x_min:x_max]
        temp_values = values[z_min:z_max, y_min:y_max, x_min:x_max]

        return (
            z_min,
            z_max,
            y_min,
            y_max,
            x_min,
            x_max,
            temp_region,
            temp_values,
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
        return (
            temp_mask_open[0, :, :].any()
            or temp_mask_open[-1, :, :].any()
            or temp_mask_open[:, 0, :].any()
            or temp_mask_open[:, -1, :].any()
            or temp_mask_open[:, :, 0].any()
            or temp_mask_open[:, :, -1].any()
        )

    def _mask_ok(
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
            return False
        if check.sum() > 1:  # More than one center inside the region of interest
            return False
        if self._touch_edges(temp_mask_open):  # Region touches the border of the volume
            return False
        return True

    def _segment(self, regions, values, centers):
        plt.figure()  # Debug
        labels = np.zeros_like(regions)
        start = 1
        for lbl in range(start, regions.max() + start):
            print(f"Segment {self.ch_id}: {lbl:3d}/{regions.max():3d}")
            # Create a mask that isolates just the region of interest (ones)
            # everything else is zero.
            current_region = regions == lbl
            # Find the smallest volumes that include the region of interest
            (
                z_min,
                z_max,
                y_min,
                y_max,
                x_min,
                x_max,
                temp_region,
                temp_values,
            ) = self._temp_masks(values, current_region)
            threshold_min = self._find_min_threshold(start, temp_region, temp_values)
            temp_mask = np.zeros_like(temp_region, dtype="bool")
            temp_mask[temp_region] = temp_values[temp_region] >= threshold_min
            #! Make sure that, after opening, you are not creating more than one
            #! connected component. In which case, decide what to do.
            temp_mask_open = ski_mor.opening(temp_mask, footprint=ski_mor.ball(5)[2::3])
            #!
            # Verify that the mask can be added to the labels
            if self._mask_ok(
                temp_mask_open, centers, z_min, z_max, y_min, y_max, x_min, x_max
            ):
                labels[
                    z_min:z_max, y_min:y_max, x_min:x_max
                ] += lbl * temp_mask_open.astype("uint16")

            print("")
        plt.show()  # Debug
        return labels

    def segment(self, labels, values, centers):
        print(f"Segmenting {self.ch_id}")
        labels = os_utils.store_output(
            self._segment,
            filename_root=self.filename_root,
            ch_id=self.ch_id,
            suffix="labels",
            regions=labels,
            values=values,
            centers=centers,
        )
        print("done!")
        return labels


class NucleiSegmentation(Segmentation):
    def __init__(self, filename_root, ch_id):
        super().__init__(filename_root, ch_id)

    def _cost(self, t, temp_region, temp_values):
        msk = np.zeros_like(temp_region, dtype="bool")
        msk[temp_region] = temp_values[temp_region] >= t

        out_idx = np.logical_and(~msk, temp_region)
        out_variance = self._evaluate_variance_in_volume(out_idx, temp_values)

        in_idx = np.logical_and(msk, temp_region)
        in_variance = self._evaluate_variance_in_volume(in_idx, temp_values)

        # surface_idx = np.logical_xor(
        #     ski_mor.binary_dilation(msk, footprint=ski_mor.ball(1)), msk
        # )
        # surface = surface_idx.sum()

        # volume = in_idx.sum()
        # volume_val = temp_values[in_idx].sum()

        table = ski_mea.regionprops_table(ski_mea.label(msk))
        objects = len(table["label"])

        return (
            # out_mean_var + in_mean_var - surface + (0 if (objects == 1) else np.inf),
            out_variance + in_variance + (0 if (objects == 1) else np.inf),
            objects,
        )


class CytoplasmSegmentation(Segmentation):
    def __init__(self, filename_root, ch_id):
        super().__init__(filename_root, ch_id)

    def _cost(self, t, temp_mask, temp_region, temp_values):
        pass
