# coding=utf-8

__author__ = "Sebastian Ahn"
__doc__ = """\
MakePaddedMask
==============
**MakePaddedMask** creates a rectangular mask from an irregularly shaped mask, with 
some padding around it.

For processing particular regions of interest within large images, performing the 
analysis on a specific region saves processing time by masking images using the
**MaskImage** module. 

"""

from typing import List, Tuple, Union, Dict

import numpy as np

import cellprofiler.measurement as cpmeas
import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw

MEASUREMENTS_MADE = ["Origin_X", "Origin_Y"]


def make_padded_array(
    array: np.ndarray, padding: int
) -> Dict[str, Union[np.ndarray, Tuple[int, int]]]:
    x = np.logical_or.reduce(array, axis=0)
    x0 = max(0, np.argmax(x) - padding)
    x1 = min(len(x), len(x) - np.argmax(x[::-1]) + padding)

    y = np.logical_or.reduce(array, axis=1)
    y0 = max(0, np.argmax(y) - padding)
    y1 = min(len(y), len(y) - np.argmax(y[::-1]) + padding)

    output_array = np.zeros_like(array, dtype=bool)
    output_array[y0:y1, x0:x1] = True
    return {"array": output_array, "origin": (int(x0), int(y0))}


def blend_arrays(
    image: np.ndarray,
    arrays: List[Tuple[np.ndarray, Tuple[float, ...]]],
    alpha1=0.5,
    alpha2=0.5,
) -> np.ndarray:
    for array, color in arrays:
        array_image = np.zeros_like(image)
        array_image[array] = color

        # alpha blending
        image = (image * alpha1 + array_image * alpha2 * (1.0 - alpha1)) / (
            alpha1 + alpha2 * (1.0 - alpha1)
        )

    return image


class MakePaddedMask(cpm.Module):
    module_name = "MakePaddedMask"
    category = "JonasLab Custom"
    variable_revision_number = 1

    def volumetric(self) -> bool:
        return False

    def create_settings(self) -> None:
        self.input_mask_name = cps.ObjectNameSubscriber(
            text="Select the input mask", value=cps.NONE, doc=""
        )

        self.padding = cps.Integer(text="Enter padding size (px)", value=20, doc="")

        self.output_mask_name = cps.ObjectNameProvider(
            text="Name the output mask",
            value="PaddedMask",
            doc="Enter the name of the output mask",
        )

        self.custom = cps.Binary(
            text="Use custom settings?", value="No", doc="Use custom settings"
        )

        self.divider = cps.Divider(line=True)

        self.image_name = cps.ImageNameSubscriber(
            text="Select image to overlay on",
            value=cps.LEAVE_BLANK,
            doc="""\
Choose the image_name upon which a padded mask constructed from the input_mask is laid.
""",
            can_be_blank=True,
        )

        self.input_mask_color = cps.Color(
            text="Select the input mask color",
            value="red",
            doc="Select the input mask color",
        )

        self.output_mask_color = cps.Color(
            text="Select the output mask color",
            value="blue",
            doc="Select the output mask color",
        )

    def settings(self) -> List[cps.Setting]:
        return [
            self.input_mask_name,
            self.padding,
            self.output_mask_name,
            self.custom,
            self.divider,
            self.image_name,
            self.input_mask_color,
            self.output_mask_color,
        ]

    def visible_settings(self) -> List[cps.Setting]:
        settings = [
            self.input_mask_name,
            self.padding,
            self.output_mask_name,
            self.custom,
        ]
        if self.custom.value:
            settings += [
                self.divider,
                self.image_name,
                self.input_mask_color,
                self.output_mask_color,
            ]
        return settings

    def get_measurement_columns(
        self, pipeline: cpp.Pipeline
    ) -> List[Tuple[str, str, str]]:
        columns = []
        for measurement_name in MEASUREMENTS_MADE:
            columns.append(
                (
                    self.input_mask_name.value,
                    "Metadata_Wedge_" + measurement_name,
                    cpmeas.COLTYPE_INTEGER,
                )
            )
        return columns

    def get_measurements(
        self, pipeline: cpp.Pipeline, object_name: str, category: str
    ) -> List[str]:
        if category == "Wedge" and self.get_categories(pipeline, object_name):
            return MEASUREMENTS_MADE
        else:
            return []

    def get_categories(self, pipeline: cpp.Pipeline, object_name: str) -> List[str]:
        if self.output_mask_name.value == object_name:
            return ["Wedge"]
        return []

    def run(self, workspace: cpw.Workspace) -> None:
        input_mask_name: str = self.input_mask_name.value
        input_mask: np.ndarray = workspace.get_objects(input_mask_name).segmented
        input_mask = input_mask.astype(dtype=bool)

        output = make_padded_array(input_mask, padding=self.padding.value)
        output_mask = output["array"]

        output_mask_obj = cpo.Objects()
        output_mask_obj.segmented = output_mask
        workspace.object_set.add_objects(output_mask_obj, self.output_mask_name.value)

        workspace.measurements.add_measurement(
            object_name=self.output_mask_name.value,
            feature_name="Metadata_Wedge_Origin_X",
            data=[output["origin"][0]],  # not sure why this has to be a list...
        )
        workspace.measurements.add_measurement(
            object_name=self.output_mask_name.value,
            feature_name="Metadata_Wedge_Origin_Y",
            data=[output["origin"][1]],
        )

        if self.show_window:
            # Initialize image to overlay on
            if self.image_name.value == cps.LEAVE_BLANK:
                image = np.zeros(shape=input_mask.shape[:2] + (3,), dtype=float)
            else:
                image: np.ndarray = workspace.image_set.get_image(
                    self.image_name.value
                ).pixel_data

            # Construct merged image
            input_mask_color: Tuple[float, ...] = tuple(
                c / 255.0 for c in self.input_mask_color.to_rgb()
            )
            output_mask_color: Tuple[float, ...] = tuple(
                c / 255.0 for c in self.output_mask_color.to_rgb()
            )

            blended_image: np.ndarray = blend_arrays(
                image=image,
                arrays=[
                    (input_mask, input_mask_color),
                    (output_mask, output_mask_color),
                ],
            )
            workspace.display_data.blended_image = blended_image

    def display(self, workspace: cpw.Workspace, figure) -> None:
        blended_image: np.ndarray = workspace.display_data.blended_image

        figure.set_subplots((1, 1))
        figure.subplot_imshow_color(
            x=0, y=0, image=blended_image, title="Padded Mask", normalize=False
        )
