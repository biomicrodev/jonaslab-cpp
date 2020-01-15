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


import numpy

import cellprofiler.module
import cellprofiler.object
import cellprofiler.setting
import cellprofiler.workspace


def make_buffer_array(array, buffer):
    x = numpy.logical_or.reduce(array, axis=0)
    x0 = max(0, numpy.argmax(x) - buffer)
    x1 = min(len(x), len(x) - numpy.argmax(x[::-1]) + buffer)

    y = numpy.logical_or.reduce(array, axis=1)
    y0 = max(0, numpy.argmax(x) - buffer)
    y1 = min(len(y), len(y) - numpy.argmax(y[::-1]) + buffer)

    output_array = numpy.zeros_like(array, dtype=bool)
    output_array[y0:y1, x0:x1] = True
    return output_array


def blend_arrays(arrays, alpha1=0.5, alpha2=0.5):
    image = numpy.zeros(shape=arrays[0][0].shape[:2] + (3,), dtype=float)

    for array, color in arrays:
        array_image = numpy.zeros_like(image)
        array_image[array] = color

        image = (image * alpha1 + array_image * alpha2 * (1.0 - alpha1)) / (
            alpha1 + alpha2 * (1.0 - alpha1)
        )

    return image


class MakePaddedMask(cellprofiler.module.Module):
    module_name = "MakePaddedMask"
    category = "JonasLab Custom"
    variable_revision_number = 1

    def volumetric(self):
        return False

    def create_settings(self):
        self.input_mask_name = cellprofiler.setting.ImageNameSubscriber(
            text="Select the input mask", value=cellprofiler.setting.NONE, doc=""
        )

        self.buffer = cellprofiler.setting.Integer(
            text="Enter buffer size (px)", value=0, doc=""
        )

        self.output_mask_name = cellprofiler.setting.ImageNameProvider(
            text="Name the output mask",
            value="BufferMask",
            doc="Enter the name of the output mask",
        )

        self.custom = cellprofiler.setting.Binary(
            text="Use custom settings?",
            value=cellprofiler.setting.NO,
            doc="Use custom settings",
        )

        self.divider = cellprofiler.setting.Divider(line=True)

        self.input_mask_color = cellprofiler.setting.Color(
            text="Select the input mask color",
            value="red",
            doc="Select the input mask color",
        )

        self.output_mask_color = cellprofiler.setting.Color(
            text="Select the output mask color",
            value="blue",
            doc="Select the output mask color",
        )

    def settings(self):
        return [
            self.input_mask_name,
            self.buffer,
            self.output_mask_name,
            self.custom,
            self.divider,
            self.input_mask_color,
            self.output_mask_color,
        ]

    def visible_settings(self):
        settings = [self.input_mask_name, self.buffer, self.output_mask_name]
        if self.custom:
            settings += [self.divider, self.input_mask_color, self.output_mask_color]
        return settings

    def run(self, workspace):
        input_mask_name = self.input_mask_name.value
        input_mask = workspace.object_set.get_object(input_mask_name)

        output_mask = make_buffer_array(input_mask, buffer=self.buffer.value)

        output_mask_obj = cellprofiler.object.Objects()
        output_mask_obj.segmented = output_mask
        workspace.object_set.add_objects(output_mask_obj, self.output_mask_name.value)

        if self.show_window:
            input_mask_color = tuple(c / 255.0 for c in self.input_mask_color.to_rgb())
            output_mask_color = tuple(
                c / 255.0 for c in self.output_mask_color.to_rgb()
            )

            blended_image = blend_arrays(
                [(input_mask, input_mask_color), (output_mask, output_mask_color)]
            )
            workspace.display_data.blended_image = blended_image

    def display(self, workspace, figure):
        blended_image = workspace.display_data.blended_image

        figure.set_subplots((1, 1))
        figure.subplot_imshow_color(
            x=0, y=0, image=blended_image, title="Buffer Mask", normalize=False
        )
