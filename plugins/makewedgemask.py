# coding=utf-8

__author__ = "Sebastian Ahn"
__doc__ = """\
MakeWedgeMask
=============

Create a mask object of the *wedge*, a region of tissue that has experienced the 
effects of bioactive substances diffused from a well. The dimensions of the wedge mask 
are chosen such that the tissue area has definitively been affected by the drug.

The output of this module is a mask object that can be used by the MaskObjects module 
to analyze objects of interest within the mask.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============
"""

import math
from typing import List, Tuple, Dict, Union

import numpy as np
import skimage.color

import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw

DEFAULT_MASK_COLOR = "green"
MEASUREMENTS_NEEDED = ["Center_X", "Center_Y", "Well_X", "Well_Y"]
MEASUREMENTS_MADE = ["Thickness", "Span", "RadialOffset", "AngularOffset"]


def cart2pol(x: float, y: float, in_deg: bool = True) -> Tuple[float, float]:
    r = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
    theta = math.atan2(y, x)

    if in_deg:
        theta = math.degrees(theta)

    return r, theta


def _make_wedge_mask(
    dims: Tuple[int, int],  # y, x
    x: int,
    y: int,
    inner_radius: float,
    thickness: float,
    th: float,
    dth: float,
) -> np.ndarray:
    """
    This function has been somewhat optimized, so make sure to time it before
    making any changes.

    Computations are done in radians to avoid calling np.degrees.
    """
    th *= math.pi / 180
    dth *= math.pi / 180

    h, w = dims

    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx -= x
    yy -= y
    theta = np.arctan2(yy, xx)

    angle = (theta - th + math.pi * 3) % (math.pi * 2) - math.pi
    # angle = theta - th

    d2 = xx ** 2 + yy ** 2
    mask = (
        (inner_radius ** 2 <= d2)
        & (d2 <= (inner_radius + thickness) ** 2)
        & (-dth <= angle)
        & (angle <= dth)
    )
    assert mask.dtype == bool
    return mask


def make_wedge_mask(
    params: Dict[str, Union[int, float]], shape: Tuple[int, int]
) -> np.ndarray:
    required_keys = {
        "Center_X",
        "Center_Y",
        "Well_X",
        "Well_Y",
        "Thickness",
        "Span",
        "RadialOffset",
        "AngularOffset",
    }
    if set(params.keys()) != required_keys:
        missing_keys = required_keys.difference(params.keys())
        raise ValueError(
            "Unable to construct wedge! Missing parameters: " + str(missing_keys)
        )

    cx, cy = (params["Center_X"], params["Center_Y"])
    wx, wy = (params["Well_X"], params["Well_Y"])
    thickness = params["Thickness"]
    span = params["Span"]
    radial_offset = params["RadialOffset"]
    angular_offset = params["AngularOffset"]

    dx = wx - cx
    dy = wy - cy

    center_length, center_angle = cart2pol(dx, dy)

    return _make_wedge_mask(
        dims=shape,
        x=cx,
        y=cy,
        inner_radius=center_length + radial_offset,
        thickness=thickness,
        th=center_angle + angular_offset,
        dth=span / 2,
    )


def blend_image_and_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, ...],
    alpha1: float = 0.5,
    alpha2: float = 0.5,
) -> np.ndarray:
    color_mask = np.zeros_like(image)
    color_mask[mask] = color

    # Alpha blending
    return (image * alpha1 + color_mask * alpha2 * (1.0 - alpha1)) / (
        alpha1 + alpha2 * (1.0 - alpha1)
    )


class MakeWedgeMask(cpm.Module):
    module_name = "MakeWedgeMask"
    category = "JonasLab Custom"
    variable_revision_number = 1

    def volumetric(self) -> bool:
        return False

    def create_settings(self) -> None:
        module_explanation = "Creates a binary mask of the wedge."
        self.set_notes([module_explanation])

        self.wedge_mask_name = cps.ObjectNameProvider(
            text="Name the wedge mask",
            value="WedgeMask",
            doc="Enter the name of the wedge mask.",
        )

        self.image_name = cps.ImageNameProvider(
            text="Select image to overlay on",
            value="None",
            doc="""\
Choose the image_name upon which a wedge mask constructed from the given 
parameters is laid. Can be either RGB or grayscale.
""",
        )

        self.divider = cps.Divider(line=True)

        self.thickness = cps.Float(
            text="Enter thickness of wedge (um)", value=400.0, doc=""
        )

        self.span = cps.Float(text="Enter span of wedge (deg)", value=90.0, doc="")

        self.radial_offset = cps.Float(
            text="Enter radial offset",
            value=0.0,
            doc="Enter offset of the inner edge of wedge from well, in microns.",
        )

        self.angular_offset = cps.Float(
            text="Enter angular offset",
            value=0.0,
            doc="""\
Enter offset of the wedge midline from well midline, in degrees, clockwise positive.
""",
        )

        self.mask_color = cps.Color(
            text="Select wedge fill color",
            value=DEFAULT_MASK_COLOR,
            doc="""\
The wedge is outlined in this color. Only applies when the result of this 
module is visualized.""",
        )

    def settings(self) -> List[cps.Setting]:
        return [
            self.wedge_mask_name,
            self.image_name,
            self.divider,
            self.thickness,
            self.span,
            self.radial_offset,
            self.angular_offset,
            self.mask_color,
        ]

    def get_measurement_columns(
        self, pipeline: cpp.Pipeline
    ) -> List[Tuple[str, str, str]]:
        columns = []
        for measurement_name in MEASUREMENTS_MADE:
            columns.append(
                (
                    self.wedge_mask_name.value,
                    "Metadata_Wedge_" + measurement_name,
                    cpmeas.COLTYPE_FLOAT,
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
        if self.wedge_mask_name.value == object_name:
            return ["Wedge"]
        return []

    def get_mask_params(self, workspace: cpw.Workspace) -> Dict[str, Union[int, float]]:
        params = {
            "Thickness": self.thickness.value,
            "Span": self.span.value,
            "RadialOffset": self.radial_offset.value,
            "AngularOffset": self.angular_offset.value,
            "MPP": workspace.measurements.get_current_image_measurement("Metadata_MPP"),
        }
        for measurement_name in MEASUREMENTS_NEEDED:
            measurement = workspace.measurements.get_current_image_measurement(
                "Metadata_Bow_" + measurement_name
            )
            params[measurement_name] = measurement
        return params

    def save_mask_params(
        self, workspace: cpw.Workspace, params: Dict[str, float]
    ) -> None:
        for measurement_name in MEASUREMENTS_MADE:
            workspace.measurements.add_measurement(
                object_name=cpmeas.IMAGE,
                feature_name="Wedge_" + measurement_name,
                data=params[measurement_name],
                data_type=cpmeas.COLTYPE_FLOAT,
            )

    def run(self, workspace: cpw.Workspace) -> None:
        # Prepare inputs
        image_name: str = self.image_name.value
        image: cpi.Image = workspace.image_set.get_image(image_name)
        input_pixels: np.ndarray = image.pixel_data

        params = self.get_mask_params(workspace)
        self.save_mask_params(workspace, params)

        # Transform from microns to pixels before processing
        mpp = params.pop("MPP")
        params["RadialOffset"] /= mpp
        params["Thickness"] /= mpp

        # Construct wedge object and place in workspace
        mask = make_wedge_mask(params, shape=input_pixels.shape[:2])
        mask_obj = cpo.Objects()
        mask_obj.segmented = mask
        workspace.object_set.add_objects(mask_obj, self.wedge_mask_name.value)

        if self.show_window:
            # Construct merged image
            image_rgb = (
                np.copy(input_pixels)
                if image.multichannel
                else skimage.color.gray2rgb(input_pixels)
            )
            color = tuple(c / 255.0 for c in self.mask_color.to_rgb())
            blended_image = blend_image_and_mask(image_rgb, mask, color)
            workspace.display_data.blended_image = blended_image

    def display(self, workspace: cpw.Workspace, figure) -> None:
        blended_image = workspace.display_data.blended_image

        figure.set_subplots((1, 1))
        figure.subplot_imshow_color(
            x=0, y=0, image=blended_image, title="Wedge Mask", normalize=False
        )
