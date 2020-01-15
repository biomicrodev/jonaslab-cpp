# coding=utf-8

__author__ = "Sebastian Ahn"
__doc__ = """\
MakeWedgeMask
=============

Create a mask object of the *wedge*, a region of tissue that has experienced the 
effects of bioactive substances diffused from a well. The dimensions of the wedge mask 
are chosen such that the tissue area has definitively been affected by the drug.

The output of this module is a mask object that can be used by the MaskObjects module 
to count objects of interest within the mask.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============
"""

import math
from typing import List, Tuple, Dict, Union

import numpy
import skimage.color

import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.setting
import cellprofiler.workspace

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
    dims: Tuple[int, int],
    x: int,
    y: int,
    inner_radius: float,
    thickness: float,
    th: float,
    dth: float,
) -> numpy.ndarray:
    """
    This function has been somewhat optimized, so make sure to time it before
    making any changes.

    Computations are done in radians to avoid calling numpy.degrees.
    """
    th *= math.pi / 180
    dth *= math.pi / 180

    h, w = dims

    xx, yy = numpy.meshgrid(numpy.arange(w), numpy.arange(h))
    xx -= x
    yy -= y
    theta = numpy.arctan2(yy, xx)

    angle = (theta - th + math.pi * 3) % (math.pi * 2) - math.pi
    # angle = theta - th

    d2 = xx ** 2 + yy ** 2
    mask = (
        (inner_radius ** 2 <= d2)
        & (d2 <= (inner_radius + thickness) ** 2)
        & (-dth <= angle)
        & (angle <= dth)
    )
    return mask


def make_wedge_mask(
    params: Dict[str, Union[int, float]], shape: Tuple[int, int]
) -> numpy.ndarray:
    valid_keys = {
        "Center_X",
        "Center_Y",
        "Well_X",
        "Well_Y",
        "Thickness",
        "Span",
        "RadialOffset",
        "AngularOffset",
    }
    if set(params.keys()) != valid_keys:
        raise ValueError("Unable to construct wedge! Missing parameters")

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
    image: numpy.ndarray,
    mask: numpy.ndarray,
    color: Tuple[float, ...],
    alpha1: float = 0.5,
    alpha2: float = 0.5,
) -> numpy.ndarray:
    color_mask = numpy.zeros_like(image)
    color_mask[mask] = color

    # Alpha blending
    return (image * alpha1 + color_mask * alpha2 * (1.0 - alpha1)) / (
        alpha1 + alpha2 * (1.0 - alpha1)
    )


class MakeWedgeMask(cellprofiler.module.Module):
    module_name = "MakeWedgeMask"
    category = "JonasLab Custom"
    variable_revision_number = 1

    def volumetric(self) -> bool:
        return False

    def create_settings(self) -> None:
        module_explanation = "Creates a binary mask of the wedge."
        self.set_notes([module_explanation])

        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            text="Select image to create mask for",
            value=cellprofiler.setting.NONE,
            doc="""\
Choose the image_name upon which a wedge mask constructed from the given 
parameters is laid. Can be either RGB or grayscale.
""",
        )

        self.wedge_mask_name = cellprofiler.setting.ObjectNameProvider(
            text="Name the wedge mask",
            value="WedgeMask",
            doc="Enter the name of the wedge mask.",
        )

        self.divider = cellprofiler.setting.Divider(line=True)

        self.thickness = cellprofiler.setting.Float(
            text="Enter thickness of wedge (um)", value=400.0, doc=""
        )

        self.span = cellprofiler.setting.Float(
            text="Enter span of wedge (deg)", value=90.0, doc=""
        )

        self.radial_offset = cellprofiler.setting.Float(
            text="Enter radial offset",
            value=0.0,
            doc="Enter offset of the inner edge of wedge from well, in microns.",
        )

        self.angular_offset = cellprofiler.setting.Float(
            text="Enter angular offset",
            value=0.0,
            doc="""\
Enter offset of the wedge midline from well midline, in degrees, clockwise positive.
""",
        )

        self.mask_color = cellprofiler.setting.Color(
            text="Select wedge fill color",
            value=DEFAULT_MASK_COLOR,
            doc="""\
The wedge is outlined in this color. Only applies when the result of this 
module is visualized.""",
        )

    def settings(self) -> List[cellprofiler.setting.Setting]:
        return [
            self.image_name,
            self.wedge_mask_name,
            self.divider,
            self.thickness,
            self.span,
            self.radial_offset,
            self.angular_offset,
            self.mask_color,
        ]

    def get_measurement_columns(
        self, pipeline: cellprofiler.pipeline.Pipeline
    ) -> List[Tuple[str, str, str]]:
        return [
            (
                cellprofiler.measurement.IMAGE,
                "Metadata_Wedge_" + measurement_name,
                cellprofiler.measurement.COLTYPE_FLOAT,
            )
            for measurement_name in MEASUREMENTS_MADE
        ]

    def get_mask_params(
        self, workspace: cellprofiler.workspace.Workspace
    ) -> Dict[str, Union[int, float]]:
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
        self, workspace: cellprofiler.workspace.Workspace, params: Dict[str, float]
    ):
        for measurement_name in MEASUREMENTS_MADE:
            workspace.measurements.add_measurement(
                object_name=cellprofiler.measurement.IMAGE,
                feature_name="Wedge_" + measurement_name,
                data=params[measurement_name],
                data_type=cellprofiler.measurement.COLTYPE_FLOAT,
            )

    def run(self, workspace: cellprofiler.workspace.Workspace) -> None:
        # Prepare inputs
        image_name: str = self.image_name.value
        image: cellprofiler.image.Image = workspace.image_set.get_image(image_name)
        input_pixels: numpy.ndarray = image.pixel_data

        params = self.get_mask_params(workspace)
        self.save_mask_params(workspace, params)

        # Transform from microns to pixels before processing
        mpp = params.pop("MPP")
        params["RadialOffset"] /= mpp
        params["Thickness"] /= mpp

        # Construct wedge object and place in workspace
        mask = make_wedge_mask(params, shape=input_pixels.shape[:2])
        mask_obj = cellprofiler.object.Objects()
        mask_obj.segmented = mask
        workspace.object_set.add_objects(mask_obj, self.wedge_mask_name.value)

        if self.show_window:
            # Construct merged image
            image_rgb = (
                numpy.copy(input_pixels)
                if image.multichannel
                else skimage.color.gray2rgb(input_pixels)
            )
            color = tuple(c / 255.0 for c in self.mask_color.to_rgb())
            blended_image = blend_image_and_mask(image_rgb, mask, color)
            workspace.display_data.blended_image = blended_image

    def display(self, workspace: cellprofiler.workspace.Workspace, figure) -> None:
        blended_image = workspace.display_data.blended_image

        figure.set_subplots((1, 1))
        figure.subplot_imshow_color(
            x=0, y=0, image=blended_image, title="Wedge Mask", normalize=False
        )
