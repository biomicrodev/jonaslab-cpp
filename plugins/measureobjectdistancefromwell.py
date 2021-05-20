# coding=utf-8

__author__ = "Sebastian Ahn"
__doc__ = """\
MeasureObjectDistanceFromWell
=============================

**MeasureObjectDistanceFromWell** calculates the distance in *pixels* between objects 
and the well.

Each object's centroid coordinates in 2D Euclidean space are transformed into polar 
coordinates using the well center as the origin, and the well direction as 0 rad. Note 
that because the origin of images has historically been located at the upper left 
corner of the image, 
"""

import math
from typing import List, Tuple

import numpy as np

import cellprofiler.measurement as cpmeas
import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.workspace as cpw


class MeasureObjectDistanceFromWell(cpm.Module):
    module_name = "MeasureObjectDistanceFromWell"
    category = "JonasLab Custom"
    variable_revision_number = 1

    def volumetric(self) -> bool:
        return False

    def create_settings(self) -> None:
        self.crop_mask_name = cps.ObjectNameSubscriber(
            text="Select the cropping mask",
            value="None",
            doc="",
            can_be_blank=True,
        )

        self.object_groups = []
        self.add_object_group(can_remove=False)
        self.divider = cps.Divider(line=True)
        self.add_object_group_button = cps.DoSomething(
            text="", label="Add another object", callback=self.add_object_group
        )

    def settings(self) -> List[cps.Setting]:
        return [self.crop_mask_name] + [group.name for group in self.object_groups]

    def visible_settings(self) -> List[cps.Setting]:
        settings = [self.crop_mask_name]
        for group in self.object_groups:
            settings += group.visible_settings()
        settings += [self.add_object_group_button, self.divider]
        return settings

    def validate_module(self, pipeline: cpp.Pipeline) -> None:
        objects = set()
        for group in self.object_groups:
            if group.name.value in objects:
                raise cps.ValidationError(
                    "{group_name} has already been selected".format(
                        group_name=group.name.value
                    ),
                    group.name,
                )
            objects.add(group.name.value)

    def add_object_group(self, can_remove: bool = True) -> None:
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))

        group.append(
            "name",
            cps.ObjectNameSubscriber(
                text="Select objects to measure",
                value="None",
                doc="Select the objects that you want to measure.",
            ),
        )

        if can_remove:
            group.append(
                "remove",
                cps.RemoveSettingButton(
                    text="",
                    label="Remove this object",
                    list=self.object_groups,
                    entry=group,
                ),
            )

        self.object_groups.append(group)

    # def prepare_run(self, workspace: cpw.Workspace) -> bool:
    #     for measurement_name in [
    #         "Metadata_Radius",
    #         "WellDistance_RadialDistance",
    #         "WellDistance_AngularDistance",
    #     ]:
    #         workspace.measurements.add_image_measurement(
    #             feature_name=measurement_name, data=0.0
    #         )
    #
    #     return True

    def get_measurement_columns(
        self, pipeline: cpp.Pipeline
    ) -> List[Tuple[str, str, str]]:
        columns = []
        for object_name in [object_group.name for object_group in self.object_groups]:
            for measurement_name in ["RadialDistance", "AngularDistance"]:
                columns += [
                    (
                        object_name.value,
                        "WellDistance_" + measurement_name,
                        cpmeas.COLTYPE_FLOAT,
                    )
                ]

        columns += [
            (
                cpmeas.IMAGE,
                "Metadata_Radius",
                cpmeas.COLTYPE_FLOAT,
            )
        ]
        return columns

    # def get_measurements(
    #     self, pipeline: cpp.Pipeline, object_name: str, category: str
    # ) -> List[str]:
    #     print("get_measurements")
    #     # Doesn't actually run...
    #     if category == "WellDistance" and self.get_categories(pipeline, object_name):
    #         return ["AngularDistance", "RadialDistance"]
    #
    #     return []

    # def get_categories(
    #     self, pipeline: cpp.Pipeline, object_name: str
    # ) -> List[str]:
    #     for object_name_var in [
    #         object_group.name for object_group in self.object_groups
    #     ]:
    #         if object_name_var.value == object_name:
    #             return ["WellDistance"]
    #     return []

    def run(self, workspace: cpw.Workspace) -> None:
        if self.crop_mask_name.value == "Leave blank":
            origin_x = 0
            origin_y = 0
        else:
            crop_mask: np.ndarray = workspace.object_set.get_objects(
                self.crop_mask_name.value
            ).segmented
            origin_x = np.logical_or.reduce(crop_mask, axis=0).argmax()
            origin_y = np.logical_or.reduce(crop_mask, axis=1).argmax()

        center_x = workspace.measurements.get_current_image_measurement(
            "Metadata_Bow_Center_X"
        )
        center_y = workspace.measurements.get_current_image_measurement(
            "Metadata_Bow_Center_Y"
        )
        well_x = workspace.measurements.get_current_image_measurement(
            "Metadata_Bow_Well_X"
        )
        well_y = workspace.measurements.get_current_image_measurement(
            "Metadata_Bow_Well_Y"
        )
        mpp = workspace.measurements.get_current_image_measurement("Metadata_MPP")

        # translate if cropped
        center_x -= origin_x
        center_y -= origin_y
        well_x -= origin_x
        well_y -= origin_y

        # Compute polar origin
        radius = math.sqrt((well_y - center_y) ** 2 + (well_x - center_x) ** 2)
        radius *= mpp
        angle = math.atan2((well_y - center_y), (well_x - center_x))

        for object_name in [obj.name.value for obj in self.object_groups]:
            objects: cpo.Objects = workspace.object_set.get_objects(object_name)
            # Compute delta in polar coordinates
            centroids: np.ndarray = objects.center_of_mass()  # y, x

            if centroids.size == 0:
                continue

            dy = centroids[:, 0] - center_y
            dx = centroids[:, 1] - center_x
            radial_dist = np.sqrt(dy ** 2 + dx ** 2)
            # everything in pixels up until now
            radial_dist *= mpp
            angular_dist = (np.arctan2(dy, dx) - angle + math.pi) % (
                2 * math.pi
            ) - math.pi

            workspace.measurements.add_image_measurement(
                feature_name="Metadata_Radius", data=radius
            )

            workspace.add_measurement(
                object_name=object_name,
                feature_name="WellDistance_RadialDistance",
                data=radial_dist,
            )
            workspace.add_measurement(
                object_name=object_name,
                feature_name="WellDistance_AngularDistance",
                data=angular_dist,
            )
