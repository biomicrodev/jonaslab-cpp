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

import numpy

import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.setting
import cellprofiler.workspace


class MeasureObjectDistanceFromWell(cellprofiler.module.Module):
    module_name = "MeasureObjectDistanceFromWell"
    category = "JonasLab Custom"
    variable_revision_number = 1

    def create_settings(self) -> None:
        self.object_groups = []
        self.add_object_group(can_remove=False)
        self.divider = cellprofiler.setting.Divider(line=True)
        self.add_object_group_button = cellprofiler.setting.DoSomething(
            text="", label="Add another object", callback=self.add_object_group
        )

    def settings(self) -> List[cellprofiler.setting.Setting]:
        return [group.name for group in self.object_groups]

    def visible_settings(self) -> List[cellprofiler.setting.Setting]:
        settings = []
        for group in self.object_groups:
            settings += group.visible_settings()
        settings += [self.add_object_group_button, self.divider]
        return settings

    def validate_module(self, pipeline: cellprofiler.pipeline.Pipeline) -> None:
        objects = set()
        for group in self.object_groups:
            if group.name.value in objects:
                raise cellprofiler.setting.ValidationError(
                    "{group_name} has already been selected".format(
                        group_name=group.name.value
                    ),
                    group.name,
                )
            objects.add(group.name.value)

    def add_object_group(self, can_remove: bool = True) -> None:
        group = cellprofiler.setting.SettingsGroup()
        if can_remove:
            group.append("divider", cellprofiler.setting.Divider(line=False))

        group.append(
            "name",
            cellprofiler.setting.ObjectNameSubscriber(
                text="Select objects to measure",
                value=cellprofiler.setting.NONE,
                doc="Select the objects that you want to measure.",
            ),
        )

        if can_remove:
            group.append(
                "remove",
                cellprofiler.setting.RemoveSettingButton(
                    text="",
                    label="Remove this object",
                    list=self.object_groups,
                    entry=group,
                ),
            )

        self.object_groups.append(group)

    def prepare_run(self, workspace: cellprofiler.workspace.Workspace) -> bool:
        for measurement_name in ["RadialDistance", "AngularDistance"]:
            workspace.measurements.add_image_measurement(
                feature_name=measurement_name,
                data=0.0,
                data_type=cellprofiler.measurement.COLTYPE_FLOAT,
            )

        return True

    # def get_categories(
    #     self, pipeline: cellprofiler.pipeline.Pipeline, object_name: str
    # ) -> List[str]:
    #     return ["WellDistance"]

    # def get_measurements(
    #     self, pipeline: cellprofiler.pipeline.Pipeline, object_name: str, category: str
    # ) -> List:
    #     if category == "WellDistance" and self.get_categories(pipeline, object_name):
    #         return ["AngularDistance", "RadialDistance"]
    #
    #     return []

    def get_measurement_columns(
        self, pipeline: cellprofiler.pipeline.Pipeline
    ) -> List[Tuple[str, str, str]]:
        columns = []
        for object_name in [object_group.name for object_group in self.object_groups]:
            print(object_name.value)
            for measurement_name in ["RadialDistance", "AngularDistance"]:
                columns += [
                    (
                        object_name.value,
                        measurement_name,
                        cellprofiler.measurement.COLTYPE_FLOAT,
                    )
                ]
        return columns

    def run(self, workspace: cellprofiler.workspace.Workspace) -> None:
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

        # Compute polar origin
        length = math.sqrt(
            (well_x ** 2 - center_x ** 2) + (well_y ** 2 - center_y ** 2)
        )
        angle = math.atan2((well_y - center_y), (well_x - center_x))

        for object_name in [obj.name.value for obj in self.object_groups]:
            objects: cellprofiler.object.Objects = workspace.object_set.get_objects(
                object_name
            )
            # Compute delta in polar coordinates
            centroids: numpy.ndarray = objects.center_of_mass()  # y, x
            dy = centroids[:, 0] - center_y
            dx = centroids[:, 1] - center_x
            radial_dist = numpy.sqrt(dy ** 2 + dx ** 2) - length
            angular_dist = numpy.arctan2(dy, dx) - angle

            workspace.add_measurement(
                object_name=object_name, feature_name="RadialDistance", data=radial_dist
            )
            workspace.add_measurement(
                object_name=object_name,
                feature_name="AngularDistance",
                data=angular_dist,
            )

    def volumetric(self) -> bool:
        return False
