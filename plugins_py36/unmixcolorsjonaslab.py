# coding=utf-8

__author__ = "Sebastian Ahn"
__doc__ = """\
UnmixColorsJonasLab
===================
**UnmixColorsJonasLab** creates an image for each dye from brightfield images 
containing more than one dye.

Spectral deconvolution depends on the characteristics of the microscope lamp. The 
default coefficients provided in CellProfiler's UnmixColors module may not always apply 
to all microscopes. In order to profile each microscope accurately, for each stain 
used, a slide with just that stain applied must be captured by the microscope. In the 
case of IHC-based dyes (e.g. DAB, as opposed to purely biochemical dyes such as DAPI or 
Hoechst), a good control tissue with an abundant biomarker should be used. 

 
References
^^^^^^^^^^

-  Ruifrok AC, Johnston DA. (2001) “Quantification of histochemical
   staining by color deconvolution.” *Analytical & Quantitative Cytology
   & Histology*, 23: 291-299.

See also **UnmixColors**.

"""


import numpy

import cellprofiler.gui.help.content
import cellprofiler.image
import cellprofiler.module
import cellprofiler.preferences
import cellprofiler.setting
import cellprofiler.workspace

MICROSCOPE_MIT = "MIT-KI - Aperio Slide Scanner"
MICROSCOPE_BWH = "BWH-SHC - Aperio Slide Scanner"
MICROSCOPE_ECHO = "Echo Revolve"
MICROSCOPES = [MICROSCOPE_MIT, MICROSCOPE_BWH, MICROSCOPE_ECHO]

CHOICE_HEMATOXYLIN = "Hematoxylin"
ST_HEMATOXYLIN = {
    MICROSCOPE_MIT: (0.644, 0.717, 0.267),
    MICROSCOPE_BWH: (0.644, 0.717, 0.267),
    MICROSCOPE_ECHO: (0.644, 0.717, 0.267),
}

CHOICE_EOSIN = "Eosin"
ST_EOSIN = {
    MICROSCOPE_MIT: (0.093, 0.954, 0.283),
    MICROSCOPE_BWH: (0.093, 0.954, 0.283),
    MICROSCOPE_ECHO: (0.093, 0.954, 0.283),
}

CHOICE_DAB = "DAB"
ST_DAB = {
    MICROSCOPE_MIT: (0.268, 0.570, 0.776),
    MICROSCOPE_BWH: (0.268, 0.570, 0.776),
    MICROSCOPE_ECHO: (0.268, 0.570, 0.776),
}

CHOICE_CUSTOM = "Custom"

STAIN_DICTIONARY = {
    CHOICE_DAB: ST_DAB,
    CHOICE_EOSIN: ST_EOSIN,
    CHOICE_HEMATOXYLIN: ST_HEMATOXYLIN,
}

STAINS_BY_POPULARITY = (CHOICE_HEMATOXYLIN, CHOICE_EOSIN, CHOICE_DAB)


def estimate_absorbance():
    """Load an image and use it to estimate the absorbance of a stain

    Returns a 3-tuple of the R/G/B absorbances
    """
    from cellprofiler.modules.loadimages import LoadImagesImageProvider
    from scipy.linalg import lstsq
    import wx

    def compute_absorbance(image_filepath):
        lip = LoadImagesImageProvider(
            name="dummy", pathname="", filename=image_filepath
        )
        image = lip.provide_image(image_set=None).pixel_data
        if image.ndim < 3:
            wx.MessageBox(
                message="You must calibrate the absorbance using a color image",
                caption="Error: not color image",
                style=wx.OK | wx.ICON_ERROR,
            )
            return None
        eps = 1.0 / 256.0 / 2.0
        log_image = numpy.log(image + eps)
        data = [-log_image[:, :, i].flatten() for i in range(3)]
        sums = [numpy.sum(x) for x in data]
        order = numpy.lexsort([sums])
        strongest = data[order[-1]][:, numpy.newaxis]
        absorbances = [lstsq(strongest, d)[0][0] for d in data]
        absorbances = numpy.array(absorbances)
        return absorbances / numpy.sqrt(numpy.sum(absorbances ** 2))

    with wx.FileDialog(
        parent=None,
        message="Choose reference image",
        wildcard="""\
Image file (*.tif, *.tiff, *.bmp, *.png, *.gif, *.jpg)|
*.tif;*.tiff;*.bmp;*.png;*.gif;*.jpg""",
        defaultDir=cellprofiler.preferences.get_default_image_directory(),
    ) as dialog:
        result = dialog.ShowModal()
        if result == wx.ID_OK:
            return compute_absorbance(image_filepath=dialog.Path)

    return None


def get_absorbances(microscope, output):
    """Given one of the outputs, return the red, green and blue absorbance"""
    if output.stain_choice == CHOICE_CUSTOM:
        result = (
            output.red_absorbance.value,
            output.green_absorbance.value,
            output.blue_absorbance.value,
        )
    else:
        result = STAIN_DICTIONARY[output.stain_choice.value][microscope]
    result = numpy.array(result)
    result /= numpy.sqrt(numpy.sum(result ** 2))  # normalize
    return result


def get_inverse_absorbances(microscope, outputs):
    absorbances = numpy.array([get_absorbances(microscope, o) for o in outputs])
    return numpy.linalg.pinv(absorbances)


def deconvolve(image, inv_abs):
    eps = 1.0 / 256.0 / 2.0
    image += eps
    log_image = numpy.log(image)
    scaled_image = log_image * inv_abs[numpy.newaxis, numpy.newaxis, :]
    image = numpy.exp(numpy.sum(scaled_image, axis=2))
    image -= eps

    image[image < 0] = 0
    image[image > 1] = 1
    image = 1 - image
    return image


class UnmixColorsJonasLab(cellprofiler.module.Module):
    module_name = "UnmixColorsJonasLab"
    category = "JonasLab Custom"
    variable_revision_number = 1

    def create_settings(self):
        self.microscope = cellprofiler.setting.Choice(
            text="Select the microscope",
            choices=MICROSCOPES,
            value=cellprofiler.setting.NONE,
            doc="""Select the microscope that was used to capture the image.""",
        )

        self.divider = cellprofiler.setting.Divider(line=True)

        self.outputs = []
        self.stain_count = cellprofiler.setting.HiddenCount(
            sequence=self.outputs, text="Stain count"
        )

        self.input_image_name = cellprofiler.setting.ImageNameSubscriber(
            text="Select the input color image",
            value=cellprofiler.setting.NONE,
            doc="""\
Choose the name of the histologically stained color image
loaded or created by some prior module.""",
        )

        self.add_image(False)

        self.add_image_button = cellprofiler.setting.DoSomething(
            text="",
            label="Add another stain",
            callback=self.add_image,
            doc="""\
Press this button to add another stain to the list.

You will be able to name the image produced and to either pick
the stain from a list of pre-calibrated stains or to enter
custom values for the stain's red, green and blue absorbance.
            """,
        )

    def add_image(self, can_remove=True):
        group = cellprofiler.setting.SettingsGroup()
        group.can_remove = can_remove
        if can_remove:
            group.append("divider", cellprofiler.setting.Divider(line=True))
        idx = len(self.outputs)
        default_name = STAINS_BY_POPULARITY[idx % len(STAINS_BY_POPULARITY)]
        default_name = default_name.replace(" ", "")

        group.append(
            "image_name",
            cellprofiler.setting.ImageNameProvider(
                text="Name the output image",
                value=default_name,
                doc="""\
Use this setting to name one of the images produced by the
module for a particular stain. The image can be used in
subsequent modules in the pipeline.
""",
            ),
        )

        choices = list(sorted(STAIN_DICTIONARY.keys())) + [CHOICE_CUSTOM]
        assert default_name in choices

        group.append(
            "stain_choice",
            cellprofiler.setting.Choice(
                text="Stain",
                choices=choices,
                value=default_name,
                doc="""\
Use this setting to choose the absorbance values for a particular stain.

The stains are:

|Unmix_image0|

(Information taken from `here`_,
`here <http://en.wikipedia.org/wiki/Staining>`__, and
`here <http://stainsfile.info>`__.)
You can choose *{CHOICE_CUSTOM}* and enter your custom values for the
absorbance (or use the estimator to determine values from single-stain
images).

.. _here: http://en.wikipedia.org/wiki/Histology#Staining
.. |Unmix_image0| image:: {HELP_CONTENT}

""".format(
                    CHOICE_CUSTOM=CHOICE_CUSTOM,
                    HELP_CONTENT=cellprofiler.gui.help.content.image_resource(
                        "UnmixColors.png"
                    ),
                ),
            ),
        )

        group.append(
            "red_absorbance",
            cellprofiler.setting.Float(
                text="Red absorbance",
                value=0.5,
                minval=0,
                maxval=1,
                doc="""\
*(Used only if "{CHOICE_CUSTOM}" is selected for the stain)*

The red absorbance setting estimates the dye’s absorbance of light in
the red channel.You should enter a value between 0 and 1 where 0 is no
absorbance and 1 is complete absorbance. You can use the estimator to
calculate this value automatically.
""".format(
                    CHOICE_CUSTOM=CHOICE_CUSTOM
                ),
            ),
        )

        group.append(
            "green_absorbance",
            cellprofiler.setting.Float(
                text="Green absorbance",
                value=0.5,
                minval=0,
                maxval=1,
                doc="""\
*(Used only if "{CHOICE_CUSTOM}" is selected for the stain)*

The green absorbance setting estimates the dye’s absorbance of light in
the green channel. You should enter a value between 0 and 1 where 0 is
no absorbance and 1 is complete absorbance. You can use the estimator to
calculate this value automatically.
""".format(
                    CHOICE_CUSTOM=CHOICE_CUSTOM
                ),
            ),
        )

        group.append(
            "blue_absorbance",
            cellprofiler.setting.Float(
                text="Blue absorbance",
                value=0.5,
                minval=0,
                maxval=1,
                doc="""\
*(Used only if "{CHOICE_CUSTOM}" is selected for the stain)*

The blue absorbance setting estimates the dye’s absorbance of light in
the blue channel. You should enter a value between 0 and 1 where 0 is no
absorbance and 1 is complete absorbance. You can use the estimator to
calculate this value automatically.
""".format(
                    CHOICE_CUSTOM=CHOICE_CUSTOM
                ),
            ),
        )

        def on_estimate():
            result = estimate_absorbance()
            if result is not None:
                (
                    group.red_absorbance.value,
                    group.green_absorbance.value,
                    group.blue_absorbance.value,
                ) = result

        group.append(
            "estimator_button",
            cellprofiler.setting.DoSomething(
                text="Estimate absorbance from image",
                label="Estimate",
                callback=on_estimate,
                doc="""\
Press this button to load an image of a sample stained only with the dye
of interest. **UnmixColors** will estimate appropriate red, green and
blue absorbance values from the image.
""",
            ),
        )

        if can_remove:
            group.append(
                "remover",
                cellprofiler.setting.RemoveSettingButton(
                    text="", label="Remove this image", list=self.outputs, entry=group
                ),
            )
        self.outputs.append(group)

    def prepare_settings(self, setting_values):
        stain_count = int(setting_values[1])
        if len(self.outputs) > stain_count:
            del self.outputs[stain_count:]
        while len(self.outputs) < stain_count:
            self.add_image()

    def settings(self):
        """The settings as saved to or loaded from the pipeline"""
        result = [self.microscope, self.stain_count, self.input_image_name]
        for output in self.outputs:
            result += [
                output.image_name,
                output.stain_choice,
                output.red_absorbance,
                output.green_absorbance,
                output.blue_absorbance,
            ]
        return result

    def visible_settings(self):
        """The settings visible to the user"""
        result = [self.microscope, self.divider, self.input_image_name]
        for output in self.outputs:
            if output.can_remove:
                result += [output.divider]
            result += [output.image_name, output.stain_choice]
            if output.stain_choice == CHOICE_CUSTOM:
                result += [
                    output.red_absorbance,
                    output.green_absorbance,
                    output.blue_absorbance,
                    output.estimator_button,
                ]
            if output.can_remove:
                result += [output.remover]
        result += [self.add_image_button]
        return result

    def run(self, workspace):
        """
        Unmix the colors on an image in the image set
        """
        input_image_name = self.input_image_name.value
        input_image = workspace.image_set.get_image(input_image_name, must_be_rgb=True)
        if self.show_window:
            workspace.display_data.input_image = input_image.pixel_data
            workspace.display_data.outputs = {}

        inv_absorbances = get_inverse_absorbances(self.microscope.value, self.outputs)
        for i, output in enumerate(self.outputs):
            inv_absorbance = inv_absorbances[:, i].flatten()
            output_pixels = deconvolve(input_image.pixel_data, inv_absorbance)

            image_name = output.image_name.value
            output_image = cellprofiler.image.Image(
                image=output_pixels, parent_image=input_image
            )
            workspace.image_set.add(image_name, output_image)
            if self.show_window:
                workspace.display_data.outputs[image_name] = output_pixels

    def display(self, workspace, figure):
        """Display all of the images in a figure"""
        figure.set_subplots((len(self.outputs) + 1, 1))
        input_image = workspace.display_data.input_image
        figure.subplot_imshow_color(
            x=0, y=0, image=input_image, title=self.input_image_name.value
        )
        ax = figure.subplot(0, 0)
        for i, output in enumerate(self.outputs):
            image_name = output.image_name.value
            pixel_data = workspace.display_data.outputs[image_name]
            figure.subplot_imshow_grayscale(
                x=i + 1, y=0, image=pixel_data, title=image_name, sharexy=ax
            )

    def volumetric(self):
        return False
