__author__ = "Sebastian Ahn"
__doc__ = """\
IdentifyBow
===========

**IdentifyBow** is an interactive method of identifying the position and orientation of 
drug release from a well. This is done by rendering a bow on the image with two 
interactive points: the center and the well.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-   *Center_X*, *Center_Y*: The position of the center of the microdevice.
-   *Well_X*, *Well_Y*: The position of the well of the microdevice.

"""

import math
from typing import Dict, Union, List, Callable, Tuple

import matplotlib.axes
import matplotlib.figure
import matplotlib.lines
import matplotlib.patches
import numpy
import wx
from PIL import Image
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas

import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.pipeline
import cellprofiler.setting
import cellprofiler.workspace

K_UP = "w"
K_DOWN = "s"
K_LEFT = "a"
K_RIGHT = "d"
ARROW_KEYS = K_UP + K_DOWN + K_LEFT + K_RIGHT
epsilon = 20  # pixels

mode2color: Dict[str, str] = {"light": "k", "dark": "w"}

MEASUREMENTS = ["Center_X", "Center_Y", "Well_X", "Well_Y"]

# Helper types
InteractorParamObj = Dict[str, Union[str, Callable, Tuple[int, int]]]


def cart2pol(x: float, y: float, in_deg: bool = True) -> Tuple[float, float]:
    r = math.sqrt(math.pow(x, 2) + math.pow(y, 2))
    theta = math.atan2(y, x)

    if in_deg:
        theta = math.degrees(theta)

    return r, theta


def pol2cart(r: float, theta: float, in_degs: bool = True) -> Tuple[float, float]:
    if in_degs:
        theta = (theta + 180) % 360 - 180
        theta = math.radians(theta)
    else:
        theta = (theta + (math.pi / 2)) % math.pi - (math.pi / 2)

    x = r * math.cos(theta)
    y = r * math.sin(theta)

    return x, y


def move(key: str, x: int, y: int) -> Tuple[int, int]:
    # image coordinates are relative to an origin that's placed at the top left
    stride = 1

    if key == K_UP:
        y -= stride
    elif key == K_DOWN:
        y += stride
    elif key == K_LEFT:
        x -= stride
    elif key == K_RIGHT:
        x += stride

    return x, y


class BaseInteractor:
    def __init__(self, axes: matplotlib.axes.Axes, **kwargs):
        """

        :param axes: Axes on which the interactor is plotted
        :param mode: "light" or "dark" mode
        """
        self.axes: matplotlib.axes.Axes = axes
        self.artists = {}

        self.__render_mode: str = kwargs.get("render_mode", "light")

    def set_render_mode(self, mode: str) -> None:
        self.__render_mode = mode

    def get_render_mode(self) -> str:
        return self.__render_mode

    render_mode = property(get_render_mode, set_render_mode)

    @property
    def color(self) -> str:
        return mode2color[self.render_mode]

    def draw_callback(self, event: MouseEvent) -> None:
        for artist in self.artists.values():
            self.axes.draw_artist(artist)

    def button_press_callback(self, event: MouseEvent) -> None:
        raise NotImplemented

    def button_release_callback(self, event: MouseEvent) -> None:
        raise NotImplemented

    def motion_notify_callback(self, event: MouseEvent) -> None:
        raise NotImplemented

    def key_press_event(self, event: MouseEvent) -> None:
        raise NotImplemented

    def key_release_event(self, event: MouseEvent) -> None:
        raise NotImplemented


class BowInteractor(BaseInteractor):
    def __init__(self, *args, **kwargs):
        """

        :param cxy: 2-tuple of xy coords for center
        :param wxy: 2-tuple of xy coords for well
        :param span: Optional float
        :param stickout: Optional int
        """
        self.id: str = kwargs.get("id")
        self.cxy: Tuple[int, int] = kwargs.get("cxy")
        self.wxy: Tuple[int, int] = kwargs.get("wxy")
        self.span: float = kwargs.get("span", 90.0)  # degrees
        self.stickout: float = kwargs.get("stickout", 1.1)  # unit-less

        super().__init__(*args, **kwargs)

        self._ind: Union[None, int] = None
        self._ind_last: Union[None, int] = None

        self.init_artists()

    def init_artists(self) -> None:
        main_arrow_props = {
            "alpha": 0.5,
            "animated": True,
            "arrowstyle": "->,head_length=10,head_width=7",
            "color": self.color,
            "linestyle": "solid",
        }

        line_props = {
            "alpha": 0.7,
            "animated": True,
            "color": self.color,
            "linestyle": "",
            "marker": "x",
            "markerfacecolor": self.color,
            "markersize": 8,
        }

        arc_props = {
            "alpha": 0.5,
            "animated": True,
            "color": self.color,
            "linestyle": "dashed",
        }

        prong_props = {
            "alpha": 0.5,
            "animated": True,
            "color": self.color,
            "linestyle": "dashed",
            "linewidth": 1,
        }

        cx, cy = self.cxy
        wx, wy = self.wxy
        dx, dy = wx - cx, wy - cy
        r, angle = cart2pol(dx, dy)
        hspan = self.span / 2

        ex, ey = pol2cart(r * self.stickout, angle)

        # main arrow
        main_arrow = matplotlib.patches.FancyArrowPatch(
            posA=self.cxy, posB=(ex + cx, ey + cy), **main_arrow_props
        )
        self.artists["main_arrow"] = main_arrow
        self.axes.add_patch(main_arrow)

        # line
        line: matplotlib.lines.Line2D = self.axes.plot(
            [cx, wx], [cy, wy], **line_props
        )[0]
        self.artists["line"] = line
        self.axes.add_line(line)

        # arc
        arc = matplotlib.patches.Arc(
            xy=self.cxy,
            width=2 * r,
            height=2 * r,
            angle=angle,
            theta1=-hspan,
            theta2=hspan,
            **arc_props
        )
        self.artists["arc"] = arc
        self.axes.add_patch(arc)

        # prongs
        r_ext = r * self.stickout
        p1_angle = angle - hspan
        p1 = {"tail": pol2cart(r, p1_angle), "head": pol2cart(r_ext, p1_angle)}
        (prong1,) = self.axes.plot(
            [cx + p1["tail"][0], cx + p1["head"][0]],
            [cy + p1["tail"][1], cy + p1["head"][1]],
            **prong_props
        )
        self.artists["prong1"] = prong1
        self.axes.add_line(prong1)

        p2_angle = angle + hspan
        p2 = {"tail": pol2cart(r, p2_angle), "head": pol2cart(r_ext, p2_angle)}
        (prong2,) = self.axes.plot(
            [cx + p2["tail"][0], cx + p2["head"][0]],
            [cy + p2["tail"][1], cy + p2["head"][1]],
            **prong_props
        )
        self.artists["prong2"] = prong2
        self.axes.add_line(prong2)

    # === COMPUTED =========================================================== #
    @property
    def angle(self) -> float:
        cx, cy = self.cxy
        wx, wy = self.wxy
        dx, dy = wx - cx, wy - cy
        _, angle = cart2pol(dx, dy)
        return angle

    # === EXPORT ============================================================= #
    def get_params(self) -> InteractorParamObj:
        return {"id": self.id, "cxy": self.cxy, "wxy": self.wxy}

    # === INTERACTION ======================================================== #
    def get_ind_under_point(self, event: MouseEvent) -> int:
        line = self.artists["line"]

        xy = numpy.asarray(line.get_xydata())
        xyt = line.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]

        d = numpy.hypot(xt - event.x, yt - event.y)
        ind_seq = numpy.nonzero(numpy.equal(d, numpy.amin(d)))[0]
        ind = ind_seq[0]

        if d[ind] >= epsilon:
            ind = None

        return ind

    def button_press_callback(self, event: MouseEvent) -> None:
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        self._ind = self.get_ind_under_point(event)
        self._ind_last = self._ind

    def button_release_callback(self, event: MouseEvent) -> None:
        if event.button != 1:
            return

        self._ind = None

    def update_all(self) -> None:
        cx, cy = self.cxy
        wx, wy = self.wxy
        dx, dy = wx - cx, wy - cy
        r, angle = cart2pol(dx, dy)

        ex, ey = pol2cart(r * self.stickout, angle)

        self.artists["main_arrow"].set_positions(self.cxy, (cx + ex, cy + ey))
        self.artists["line"].set_data([cx, wx], [cy, wy])

        arc = self.artists["arc"]
        arc.angle = angle
        arc.width = 2 * r
        arc.height = 2 * r
        arc.set_center(self.cxy)

        r_ext = r * self.stickout
        hspan = self.span / 2
        p1_angle = angle - hspan
        p1 = {"tail": pol2cart(r, p1_angle), "head": pol2cart(r_ext, p1_angle)}
        self.artists["prong1"].set_data(
            [cx + p1["tail"][0], cx + p1["head"][0]],
            [cy + p1["tail"][1], cy + p1["head"][1]],
        )

        p2_angle = angle + hspan
        p2 = {"tail": pol2cart(r, p2_angle), "head": pol2cart(r_ext, p2_angle)}
        self.artists["prong2"].set_data(
            [cx + p2["tail"][0], cx + p2["head"][0]],
            [cy + p2["tail"][1], cy + p2["head"][1]],
        )

    def motion_notify_callback(self, event: MouseEvent) -> None:
        if self._ind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        x, y = int(round(event.xdata)), int(round(event.ydata))
        if self._ind == 1:
            self._set_arrow_head(x, y)
            self.update_all()

        elif self._ind == 0:
            self._set_arrow_tail(x, y)
            self.update_all()

    def key_press_event(self, event: MouseEvent) -> None:
        if self._ind_last is None:
            return
        if event.inaxes is None:
            return

        key = event.key
        if self._ind_last == 1:
            # arrow head moved
            wx, wy = self.wxy
            wx, wy = move(key, wx, wy)
            self._set_arrow_head(wx, wy)
            self.update_all()

        elif self._ind_last == 0:
            cx, cy = self.cxy
            cx, cy = move(key, cx, cy)
            self._set_arrow_tail(cx, cy)
            self.update_all()

    def key_release_event(self, event: MouseEvent) -> None:
        if self._ind_last is None:
            return
        if event.inaxes is None:
            return

    # === ATOMIC INTERACTION ================================================= #
    def _set_arrow_head(self, x: int, y: int) -> None:
        self.wxy = x, y

    def _set_arrow_tail(self, x: int, y: int) -> None:
        self.cxy = x, y


class BaseInteractorsPanel(wx.Panel):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.interactors: List[BaseInteractor] = []
        self.factor = None
        self.image_id = None
        self.background = None

        self.BuildUI()

    def BuildUI(self):
        self.figure: matplotlib.figure.Figure = matplotlib.figure.Figure()
        self.axes: matplotlib.axes.Axes = self.figure.add_subplot(1, 1, 1)
        self.axes.set_aspect("equal")
        self.canvas = FigureCanvas(self, id=wx.ID_ANY, figure=self.figure)
        self.figure.tight_layout()

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.canvas, flag=wx.EXPAND, proportion=1)
        self.SetSizer(sizer)

        self.canvas.mpl_connect("draw_event", self.DrawCallback)
        self.canvas.mpl_connect("button_press_event", self.OnClick)
        self.canvas.mpl_connect("button_release_event", self.OnMouseButtonUp)
        self.canvas.mpl_connect("motion_notify_event", self.OnMouseMoved)
        self.canvas.mpl_connect("key_press_event", self.OnKeyPress)
        self.canvas.mpl_connect("key_release_event", self.OnKeyRelease)

    def DrawCallback(self, event: MouseEvent):
        self.background = self.canvas.copy_from_bbox(self.axes.bbox)

        for interactor in self.interactors:
            interactor.draw_callback(event)

        self.canvas.blit(self.axes.bbox)

    def OnClick(self, event: MouseEvent):
        if event.inaxes != self.axes:
            return
        if event.inaxes.get_navigate_mode() is not None:
            return

        for interactor in self.interactors:
            interactor.button_press_callback(event)

    def OnMouseButtonUp(self, event: MouseEvent):
        if event.inaxes is not None and event.inaxes.get_navigate_mode() is not None:
            return

        for interactor in self.interactors:
            interactor.button_release_callback(event)

    def OnMouseMoved(self, event: MouseEvent):
        if event.inaxes != self.axes:
            return

        self.UpdateInteractors(event)

    def OnKeyPress(self, event: MouseEvent):
        if event.inaxes != self.axes:
            return
        if event.inaxes.get_navigate_mode() is not None:
            return

        for interactor in self.interactors:
            interactor.key_press_event(event)

        self.UpdateInteractors(event)

    def OnKeyRelease(self, event: MouseEvent):
        if event.inaxes != self.axes:
            return
        if event.inaxes.get_navigate_mode() is not None:
            return

        for interactor in self.interactors:
            interactor.key_release_event(event)

        self.UpdateInteractors(event)

    def UpdateInteractors(self, event: MouseEvent):
        if self.background is not None:
            self.canvas.restore_region(self.background)
        else:
            self.background = self.canvas.copy_from_bbox(self.axes.bbox)

        for interactor in self.interactors:
            interactor.motion_notify_callback(event)
            interactor.draw_callback(event)

        self.canvas.blit(self.axes.bbox)

    def Render(self, image: Image):
        self.axes.clear()
        self.axes.imshow(image, interpolation="lanczos", vmin=0, vmax=255)


class DevicesInteractorsPanel(BaseInteractorsPanel):
    def __init__(self, parent=None):
        super().__init__(parent=parent)

    def SetInteractors(self, interactors: List[InteractorParamObj]) -> None:
        self.interactors = []
        for interactor in interactors:
            assert {"id", "cxy", "wxy", "artist"}.issubset(interactor.keys())
            artist = interactor["artist"](self.axes, **interactor)
            self.interactors.append(artist)

    def GetInteractors(self) -> List[InteractorParamObj]:
        return [interactor.get_params() for interactor in self.interactors]


class SingleBowAnnotationDialog(wx.Dialog):
    def __init__(self):
        title = "Bow Annotation"
        frame_style = (
            wx.MAXIMIZE_BOX
            | wx.MINIMIZE_BOX
            | wx.RESIZE_BORDER
            | wx.SYSTEM_MENU
            | wx.CAPTION
            | wx.CLOSE_BOX
            | wx.CLIP_CHILDREN
        )
        frame_size = 1500, 800  # width, height

        super().__init__(parent=None, title=title, style=frame_style, size=frame_size)

        self.interactorsP = DevicesInteractorsPanel(self)

        buttonSize = (70, 30)
        okBtn = wx.Button(self, id=wx.ID_OK, label="OK", size=buttonSize)
        buttonSizer = wx.BoxSizer(wx.HORIZONTAL)
        buttonSizer.Add(okBtn, flag=wx.ALL, border=5)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.interactorsP, flag=wx.EXPAND, proportion=1)
        sizer.Add(buttonSizer, flag=wx.ALL | wx.CENTER, border=5)
        self.SetSizer(sizer)

    def SetImage(self, image: Image) -> None:
        self.interactorsP.Render(image)

    def SetInteractors(self, *args, **kwargs) -> None:
        self.interactorsP.SetInteractors(*args, **kwargs)

    def GetInteractors(self) -> InteractorParamObj:
        interactors = self.interactorsP.GetInteractors()
        assert len(interactors) == 1
        return interactors[0]


class IdentifyBow(cellprofiler.module.Module):
    module_name = "IdentifyBow"
    category = "JonasLab Custom"
    variable_revision_number = 0

    def volumetric(self) -> bool:
        return False

    def create_settings(self) -> None:
        module_explanation = (
            "Construct a bow from User-supplied parameters regarding the position and "
            "orientation of drug release from a well. Usage notes: Because this module "
            "is interactive, a graphical user interface must be shown. The eye icon "
            "only toggles the visibility of the result of choosing the bow. Note that "
            "in Analysis mode, each core displays its own window."
        )
        self.set_notes([module_explanation])

        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            text="Select image to identify bow on",
            value=cellprofiler.setting.NONE,
            doc="Select image to identify bow on",
        )

        self.advanced = cellprofiler.setting.Binary(
            text="Use custom settings?",
            value=False,
            doc="Set how the bow is rendered on the image.",
        )

        self.spacer = cellprofiler.setting.Divider(line=True)

        self.span = cellprofiler.setting.Float(
            text="Set bow span", value=90.0, doc="Set the bow span, in degrees."
        )

        self.stickout = cellprofiler.setting.Float(
            text="Set bow stickout",
            value=1.3,
            doc="""\
            Set the bow stickout, as a factor of the distance between the center and 
            the well.""",
        )

    def settings(self) -> List[cellprofiler.setting.Setting]:
        return [self.image_name, self.advanced, self.spacer, self.span, self.stickout]

    def visible_settings(self) -> List[cellprofiler.setting.Setting]:
        settings = [self.image_name, self.advanced]
        if self.advanced:
            settings += [self.spacer, self.span, self.stickout]
        return settings

    def get_measurement_columns(self, pipeline: cellprofiler.pipeline.Pipeline):
        return [
            (
                cellprofiler.measurement.IMAGE,
                "Metadata_Bow_" + measurement_name,
                cellprofiler.measurement.COLTYPE_INTEGER,
            )
            for measurement_name in MEASUREMENTS
        ]

    def prepare_run(self, workspace: cellprofiler.workspace.Workspace) -> None:
        for measurement_name in ["CenterX", "CenterY", "WellX", "WellY"]:
            workspace.measurements.add_measurement(
                object_name=cellprofiler.measurement.IMAGE,
                feature_name="Wedge_" + measurement_name,
                data=0,
                data_type=cellprofiler.measurement.COLTYPE_INTEGER,
            )

    def run(self, workspace: cellprofiler.workspace.Workspace) -> None:
        image_name: str = self.image_name.value
        image: Image = workspace.image_set.get_image(image_name).pixel_data

        # get measurements
        params_pre = [
            {
                "id": "0",
                "artist": BowInteractor,
                "cxy": [100, 100],
                "wxy": [200, 200],
                "span": self.span.value,
                "stickout": self.stickout.value,
            }
        ]
        params_post: InteractorParamObj = workspace.interaction_request(
            self, image, params_pre
        )
        assert {"id", "cxy", "wxy"} == set(params_post.keys())
        # save measurements
        cx, cy = params_post["cxy"]
        wx, wy = params_post["wxy"]
        workspace.measurements.add_image_measurement(
            feature_name="Bow_Center_X", data=cx
        )
        workspace.measurements.add_image_measurement(
            feature_name="Bow_Center_Y", data=cy
        )
        workspace.measurements.add_image_measurement(feature_name="Bow_Well_X", data=wx)
        workspace.measurements.add_image_measurement(feature_name="Bow_Well_Y", data=wy)

    def handle_interaction(
        self, image: Image, bow_params_pre: InteractorParamObj
    ) -> Union[InteractorParamObj, None]:
        with SingleBowAnnotationDialog() as dialog:
            dialog.SetImage(image)
            dialog.SetInteractors(bow_params_pre)

            result = dialog.ShowModal()
            if result == wx.ID_OK:
                return dialog.GetInteractors()
            raise RuntimeError("Unable to obtain bow parameters!")
