import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from cmap import Colormap
from loguru import logger
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

from qscope.gui.util.cbar import add_colorbar
from qscope.gui.widgets import QuComboBox

mpl.use("Qt5Agg")

import qscope.server
from qscope.gui.figures.matplotlib_canvas import MplCanvas
from qscope.gui.util.error_handling import show_critial_error, show_info, show_warning


class QImageFigure(QWidget):
    def __init__(self, parent, cbar_type="horizontal"):
        super().__init__()

        self.parent = parent
        self.cbar_type = cbar_type
        self.cbar = None

        self.canvas = MplCanvas(self, width=3, height=4, dpi=100)
        self.canvas.setStatusTip(
            "Left click/drag to select AOI, right click to remove."
        )

        self.ax = self.canvas.axes
        self.fig = self.canvas.figure

        self.sig_frame = np.random.rand(512, 512) + 1
        self.sig_frame_to_plt = self.sig_frame.copy()
        self.ref_frame = np.random.rand(512, 512) + 0.8
        self.ref_frame_to_plt = self.ref_frame.copy()
        self.img = np.random.rand(512, 512)

        self.norm_type = "signal"

        self.bin_x = 1
        self.bin_y = 1

        self.init_plot(self.img)

        self.rect = None

    def init_plot(self, img=None):
        if img is not None:
            self.img = img

        # clear the figure
        self.ax.clear()
        if self.cbar is not None:
            try:
                self.cbar.remove()
                del self.cbar
                self.cbar = None
            except:
                pass

        # make the plot
        self.plot = self.ax.imshow(
            self.img,
            # cmap="viridis",
            cmap=Colormap("seaborn:mako").to_mpl(),
            origin="upper",
            extent=[0, self.img.shape[1], self.img.shape[0], 0],
        )

        self.ax.set_xlabel("X (pixels)", fontsize=9)
        self.ax.set_ylabel("Y (pixels)", fontsize=9)
        # set the ticks size
        self.ax.tick_params(axis="x", labelsize=8, direction="in")
        self.ax.tick_params(axis="y", labelsize=8, direction="in")

        # make the figure have a tight layout
        # self.fig.tight_layout()
        self.fig.get_layout_engine().set(w_pad=0, h_pad=0.2)

        # # set the axis color
        if self.cbar is None:
            if self.cbar_type == "horizontal":
                # self.cbar = self.fig.colorbar(
                #     mappable=self.plot,
                #     orientation="horizontal",
                #     location="top",
                #     aspect=25,
                # )
                self.cbar = add_colorbar(
                    self.plot,
                    self.fig,
                    self.ax,
                    aspect=25,
                    orientation="horizontal",
                    location="top",
                    labelpad=5,
                )
                self.cbar.ax.tick_params(axis="x", labelsize=8, direction="in")
                # self.cbar.set_label("Intensity (cps)", fontsize=9)
            else:
                # self.cbar = self.fig.colorbar(
                #     mappable=self.plot,
                #     orientation="vertical",
                #     location="right",
                #     aspect=25,
                # )
                # self.cbar.ax.tick_params(axis="y")
                self.cbar = add_colorbar(
                    self.plot,
                    self.fig,
                    self.ax,
                    aspect=25,
                    orientation="vertical",
                    labelpad=5,
                )
                # self.cbar.set_label("Intensity (cps)", rotation=270, fontsize=9)
                self.cbar.ax.tick_params(axis="y", labelsize=8, direction="in")

    def set_normalisation(self, norm_type):
        self.norm_type = norm_type.lower()
        if self.norm_type == "signal":
            self.img = self.sig_frame_to_plt
        elif self.norm_type == "reference":
            self.img = self.ref_frame_to_plt
        elif self.norm_type == "subtract":
            self.img = self.sig_frame_to_plt - self.ref_frame_to_plt
        elif self.norm_type == "divide":
            self.img = self.sig_frame_to_plt / self.ref_frame_to_plt
        elif self.norm_type == "normalise (%)":
            self.img = 100 * self.sig_frame_to_plt / self.ref_frame_to_plt
        else:
            self.img = self.sig_frame_to_plt

        self.update_plot()

    def update_data(self, sig_frame, ref_frame):
        # apply the requested binning
        # if self.bin_x > 1 or self.bin_y > 1:
        sig_frame_to_plt = (
            sig_frame.reshape(
                sig_frame.shape[0] // self.bin_y,
                self.bin_y,
                sig_frame.shape[1] // self.bin_x,
                self.bin_x,
            )
            .mean(3)
            .mean(1)
        )
        ref_frame_to_plt = (
            ref_frame.reshape(
                ref_frame.shape[0] // self.bin_y,
                self.bin_y,
                ref_frame.shape[1] // self.bin_x,
                self.bin_x,
            )
            .mean(3)
            .mean(1)
        )
        # update the plot extent
        self.plot.set_extent(
            [0, sig_frame_to_plt.shape[1], sig_frame_to_plt.shape[0], 0]
        )
        self.sig_frame_to_plt = sig_frame_to_plt
        self.ref_frame_to_plt = ref_frame_to_plt
        self.set_normalisation(self.norm_type)
        self.update_plot()

    def update_plot(self, img=None, *args, **kwargs):
        if img is None:
            img = self.img
        else:
            self.img = img
        self.plot.set_data(img)
        self.plot.autoscale()
        self.canvas.draw()

    def set_binning(self, bin_x, bin_y):
        self.bin_x = bin_x
        self.bin_y = bin_y
        # update the plots tick labels
        self.update_data(self.sig_frame, self.ref_frame)

    def set_cmap(self, cmap):
        self.plot.set_cmap(cmap)
        self.update_plot()


class QImageFigureOpts(QGroupBox):
    def __init__(
        self, parent, target_figure, aoi_zoom_figure=None, title="Image plot options"
    ):
        super().__init__(title, parent)
        self.parent = parent
        self.figure = target_figure
        self.aoi_zoom_figure = aoi_zoom_figure
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")

        # self.img_plot_label = QLabel("Image Plot options")
        # self.img_plot_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        # self.img_plot_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # add a dropdown menu for the frame normalisation options
        self.frame_norm_label = QLabel("Normalisation")
        self.frame_norm_dropdown = QuComboBox()
        # self.frame_norm_dropdown.setEditable(True)
        # self.frame_norm_dropdown.lineEdit().setReadOnly(True)
        # self.frame_norm_dropdown.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_norm_dropdown.addItem("Signal")
        self.frame_norm_dropdown.addItem("Reference")
        self.frame_norm_dropdown.addItem("Subtract")
        self.frame_norm_dropdown.addItem("Divide")

        # add a row for the frame normalisation options
        self.norm_row = QHBoxLayout()
        self.norm_row.addWidget(self.frame_norm_label)
        self.norm_row.addWidget(self.frame_norm_dropdown)

        # add a dropdown menu for the colourmap that change it's items based on the cmap style
        self.cmap_label = QLabel("colour map")
        self.cmap_dropdown = QuComboBox()
        # self.cmap_dropdown.setEditable(True)
        # self.cmap_dropdown.lineEdit().setReadOnly(True)
        # self.cmap_dropdown.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cmap_dropdown.addItem("seaborn:mako")
        self.cmap_dropdown.addItem("viridis")
        self.cmap_dropdown.addItem("plasma")
        self.cmap_dropdown.addItem("inferno")
        self.cmap_dropdown.addItem("magma")
        self.cmap_dropdown.addItem("cividis")
        self.cmap_dropdown.insertSeparator(6)
        self.cmap_dropdown.addItems(
            ["Greys_r", "Bone", "Blues", "Greens", "Oranges", "Reds"]
        )
        self.cmap_dropdown.insertSeparator(13)
        self.cmap_dropdown.addItems(
            [
                "PuOr",
                "RdGy",
                "RdBu",
                "Spectral",
                "coolwarm",
                "bwr",
                "seismic",
            ]
        )

        # Make the widget element
        self.cmap_row = QHBoxLayout()
        self.cmap_row.addWidget(self.cmap_label)
        self.cmap_row.addWidget(self.cmap_dropdown)

        # add a edit box for the frame number
        self.frame_idx_label = QLabel("Frame to plot:")
        self.frame_idx_input = QDoubleSpinBox()
        self.frame_idx_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_idx_input.setRange(0, 10000)
        self.frame_idx_input.setValue(0)
        self.frame_idx_input.setSingleStep(1)
        self.frame_idx_input.setDecimals(0)

        self.frame_idx_row = QHBoxLayout()
        self.frame_idx_row.addWidget(self.frame_idx_label)
        self.frame_idx_row.addWidget(self.frame_idx_input)

        # add input boxes for binning in x and y
        self.binning_x_label = QLabel("Binning x, y:")
        self.binning_x_input = QDoubleSpinBox()
        self.binning_x_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.binning_x_input.setRange(1, 8)
        self.binning_x_input.setValue(1)
        self.binning_x_input.setSingleStep(1)
        self.binning_x_input.setDecimals(0)

        self.binning_y_input = QDoubleSpinBox()
        self.binning_y_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.binning_y_input.setRange(1, 8)
        self.binning_y_input.setValue(1)
        self.binning_y_input.setSingleStep(1)
        self.binning_y_input.setDecimals(0)

        self.binning_row = QHBoxLayout()
        self.binning_row.addWidget(self.binning_x_label)
        self.binning_row.addWidget(self.binning_x_input)
        self.binning_row.addWidget(self.binning_y_input)

        # add the frame plot options to the frame plot layout
        self.layout = QVBoxLayout()
        # self.layout.addWidget(self.img_plot_label)
        self.layout.addLayout(self.norm_row)
        self.layout.addLayout(self.cmap_row)
        self.layout.addLayout(self.frame_idx_row)
        self.layout.addLayout(self.binning_row)
        self.layout.addStretch()

        self.setLayout(self.layout)

        # connect the dropdowns to the callback functions
        self.frame_norm_dropdown.currentIndexChanged.connect(self.set_normalisation)

        self.cmap_dropdown.currentIndexChanged.connect(self.select_cmap_type)

        self.frame_idx_input.valueChanged.connect(self.frame_idx_changed)

        self.binning_x_input.valueChanged.connect(self.binning_changed)
        self.binning_y_input.valueChanged.connect(self.binning_changed)

    def binning_changed(self):
        bin_x = int(self.binning_x_input.text())
        bin_y = int(self.binning_y_input.text())
        # check that the binning is valid
        if bin_x < 1 or bin_y < 1:
            show_warning(self, "Binning must be greater than 0")
            return

        # check that if data can be binned by the requested amount,
        # i.e. the data shape is divisible by the binning
        if (
            self.figure.sig_frame.shape[0] % bin_y != 0
            or self.figure.sig_frame.shape[1] % bin_x != 0
        ):
            show_warning(self, "Data shape not divisible by binning")
            return

        # update the plots binning
        self.figure.set_binning(bin_x, bin_y)
        self.aoi_zoom_figure.update_data(
            self.figure.sig_frame_to_plt, self.figure.ref_frame_to_plt
        )
        # self.img_aoi_opts.on_aoi_click(event=None)

    def frame_idx_changed(self):
        # check that the frame index is valid
        if self.frame_idx_input.value() < 0:
            return
        try:
            self.parent.connection_manager.measurement_set_frame_num(
                self.parent.meas_id, int(self.frame_idx_input.value())
            )
        except:
            logger.warning("Frame idx change invalid, resetting to zero")
            self.parent.connection_manager.measurement_set_frame_num(
                self.parent.meas_id, int(0)
            )
            self.frame_idx_input.setValue(0)

    def select_cmap_type(self):
        """Callback for the cmap_dropdown currentIndexChanged event"""

        # update the plots
        if self.cmap_dropdown.currentText() != "" or None:
            # if hasattr(self, 'figure'):
            # self.figure.set_cmap(self.cmap_dropdown.currentText())
            # if hasattr(self, 'aoi_zoom_figure'):
            # self.aoi_zoom_figure.set_cmap(self.cmap_dropdown.currentText())

            colormap = Colormap(self.cmap_dropdown.currentText())
            self.figure.set_cmap(colormap.to_mpl())
            self.aoi_zoom_figure.set_cmap(colormap.to_mpl())

    def set_normalisation(self):
        norm_type = self.frame_norm_dropdown.currentText()
        self.figure.set_normalisation(norm_type)
        self.aoi_zoom_figure.set_normalisation(norm_type)


class QImageAOIOpts(QGroupBox):
    """
    Class to hold the options for
    selecting an area of interest on a figure,
    plotting the rectangle for this region,
    and selected the zoomed in image.
    """

    def __init__(
        self, parent, target_figure, aoi_zoom_figure, title="Image AOI options"
    ):
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("QGroupBox { font-weight: bold; }")

        self.parent = parent
        self.target_figure = target_figure
        self.canvas = target_figure.canvas
        self.fig = target_figure.fig
        self.ax = self.fig.axes[0]
        self.aoi_zoom_fig = aoi_zoom_figure
        self.rect = None

        # add the gui elements
        # self.img_aoi_label = QLabel("Image AOI options")
        # self.img_aoi_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        # self.img_aoi_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create a horizontal layout for the buttons
        self.button_row = QHBoxLayout()

        self.use_aoi_button = QPushButton("Use AOI")
        self.use_aoi_button.setCheckable(True)

        self.square_aoi_button = QPushButton("Square AOI")
        self.square_aoi_button.setCheckable(True)
        self.square_aoi_button.setToolTip("Force AOI selection to be square")
        self.square_aoi_button.clicked.connect(self.on_square_button_clicked)

        self.button_row.addWidget(self.use_aoi_button)
        self.button_row.addWidget(self.square_aoi_button)

        # Current pixel
        self.aoi_pt_label = QLabel("Selected pixel x,y:")
        self.aoi_x_pt_input = QDoubleSpinBox()
        self.aoi_x_pt_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aoi_x_pt_input.setRange(1, 10000)
        self.aoi_x_pt_input.setValue(25)
        self.aoi_x_pt_input.setSingleStep(1)
        self.aoi_x_pt_input.setDecimals(0)
        self.aoi_x_pt = 25

        self.aoi_y_pt_input = QDoubleSpinBox()
        self.aoi_y_pt_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aoi_y_pt_input.setRange(1, 10000)
        self.aoi_y_pt_input.setValue(25)
        self.aoi_y_pt_input.setSingleStep(1)
        self.aoi_y_pt_input.setDecimals(0)
        self.aoi_y_pt = 25

        # Make a row that contains the x and y pts
        self.aoi_pt_row = QHBoxLayout()
        self.aoi_pt_row.addWidget(self.aoi_pt_label)
        self.aoi_pt_row.addWidget(self.aoi_x_pt_input)
        self.aoi_pt_row.addWidget(self.aoi_y_pt_input)

        # Pixel range
        self.aoi_range_label = QLabel("AOI range x,y:")
        self.aoi_x_range_input = QDoubleSpinBox()
        self.aoi_x_range_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aoi_x_range_input.setRange(1, 10000)
        self.aoi_x_range_input.setValue(20)
        self.aoi_x_range_input.setSingleStep(1)
        self.aoi_x_range_input.setDecimals(0)
        self.aoi_x_range = 20

        self.aoi_y_range_input = QDoubleSpinBox()
        self.aoi_y_range_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.aoi_y_range_input.setRange(1, 10000)
        self.aoi_y_range_input.setValue(20)
        self.aoi_y_range_input.setSingleStep(1)
        self.aoi_y_range_input.setDecimals(0)
        self.aoi_y_range = 20

        # Make a row that contains the x and y range inputs
        self.aoi_range_row = QHBoxLayout()
        self.aoi_range_row.addWidget(self.aoi_range_label)
        self.aoi_range_row.addWidget(self.aoi_x_range_input)
        self.aoi_range_row.addWidget(self.aoi_y_range_input)

        # Add drag operation variables
        self.dragging = False
        self.start_point = None
        self.current_point = None
        self.temp_rect = None

        # Add debounce timer for drawing
        self.draw_timer = QTimer()
        self.draw_timer.setSingleShot(True)
        self.draw_timer.setInterval(50)  # 50ms debounce
        self.draw_timer.timeout.connect(self.update_temp_rectangle)

        # Connect the buttons to the callback functions
        self.use_aoi_button.clicked.connect(self.use_aoi_click)
        self.aoi_x_range_input.textChanged.connect(self.aoi_range_changed)
        self.aoi_y_range_input.textChanged.connect(self.aoi_range_changed)
        self.aoi_x_pt_input.textChanged.connect(self.aoi_pt_changed)
        self.aoi_y_pt_input.textChanged.connect(self.aoi_pt_changed)

        # Connect mouse events for drag selection
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

        # Add the widgets to the layout and add a boarder to the widget
        self.layout = QVBoxLayout()
        # self.layout.addWidget(self.img_aoi_label)
        self.layout.addLayout(self.button_row)
        self.layout.addLayout(self.aoi_pt_row)
        self.layout.addLayout(self.aoi_range_row)
        self.layout.addStretch()

        self.setLayout(self.layout)

    def use_aoi_click(self):
        """Callback for the use_roi_button click event"""
        if hasattr(self.parent, "meas_id"):
            if self.use_aoi_button.isChecked():
                # change the background color of the button to blue
                self.use_aoi_button.setStyleSheet("background-color: blue")
                # send the current aoi to the server
                aoi_tuple = (
                    self.aoi_y_pt - self.aoi_y_range // 2,
                    self.aoi_x_pt - self.aoi_x_range // 2,
                    self.aoi_y_pt + self.aoi_y_range // 2,
                    self.aoi_x_pt + self.aoi_x_range // 2,
                )
                self.parent.connection_manager.measurement_set_aoi(
                    self.parent.meas_id, aoi_tuple
                )
            else:
                self.use_aoi_button.setStyleSheet("background-color: None")
                # if self.rect is not None:
                #     self.rect.remove()
                #     self.rect = None
                #     self.canvas.draw()

                # set the AOI to be the full image
                camera_shape = self.target_figure.sig_frame.shape
                # Define the AOI as the full image with the centre pixel
                aoi_tuple = (0, 0, camera_shape[1], camera_shape[0])
                # aoi_tuple = None
                self.parent.connection_manager.measurement_set_aoi(
                    self.parent.meas_id, aoi_tuple
                )

    def aoi_range_changed(self):
        logger.critical("HERE")
        try:
            self.aoi_x_range = int(self.aoi_x_range_input.text())
            self.aoi_y_range = int(self.aoi_y_range_input.text())

            # Only enforce square if values are different
            if (
                self.square_aoi_button.isChecked()
                and self.aoi_x_range != self.aoi_y_range
            ):
                size = max(self.aoi_x_range, self.aoi_y_range)
                self.aoi_x_range = size
                self.aoi_y_range = size
                self.aoi_x_range_input.setValue(size)
                self.aoi_y_range_input.setValue(size)

        except ValueError:
            self.aoi_x_range = 10
            self.aoi_y_range = 10

        if self.aoi_x_pt is not None and self.aoi_y_pt is not None:
            self.on_aoi_click(event=None)

    def aoi_pt_changed(self):
        try:
            self.aoi_x_pt = int(self.aoi_x_pt_input.text())
            self.aoi_y_pt = int(self.aoi_y_pt_input.text())
        except ValueError:
            self.aoi_x_pt = 25
            self.aoi_y_pt = 25

        if self.aoi_x_pt is not None and self.aoi_y_pt is not None:
            self.on_aoi_click(event=None)

    def on_aoi_click(self, event):
        """Handle AOI selection via click or after drag completion."""
        if event is None:
            x, y = self.aoi_x_pt, self.aoi_y_pt
            width = self.aoi_x_range
            height = self.aoi_y_range
        elif event.inaxes:
            x = int(event.xdata)
            y = int(event.ydata)
            self.x_pixel = x
            self.y_pixel = y
            width = self.aoi_x_range
            height = self.aoi_y_range

            self.aoi_x_pt_input.setValue(x)
            self.aoi_y_pt_input.setValue(y)

        # Remove existing rectangle if present
        if self.rect is not None:
            try:
                self.rect.remove()
            except ValueError:
                show_warning(self, "Rectangle already removed")

        # Create new rectangle centered on selection point
        self.rect = plt.Rectangle(
            (x - width // 2, y - height // 2),
            width,
            height,
            edgecolor="red",
            facecolor="none",
        )
        self.canvas.axes.add_patch(self.rect)
        self.canvas.draw()

        # update the roi image
        aoi_image = self.target_figure.img[
            y - self.aoi_y_range // 2 : y + self.aoi_y_range // 2,
            x - self.aoi_x_range // 2 : x + self.aoi_x_range // 2,
        ]

        # update the figure extent
        x_min = x - self.aoi_x_range // 2
        x_max = x + self.aoi_x_range // 2
        y_min = y - self.aoi_y_range // 2
        y_max = y + self.aoi_y_range // 2

        # Ensure minimum size difference to avoid singular transformation
        if x_min == x_max:
            x_max += 1
        if y_min == y_max:
            y_max += 1

        self.aoi_zoom_fig.plot.set_extent([x_min, x_max, y_max, y_min])

        # Update the region canvas
        self.aoi_zoom_fig.update_plot(img=aoi_image)

        # auto-use aoi
        if not self.use_aoi_button.isChecked():
            self.use_aoi_button.click()

    def on_mouse_press(self, event):
        """Handle mouse press event to start AOI selection."""
        if not event.inaxes:
            return

        if event.button == 1:  # Left click
            self.start_point = (int(event.xdata), int(event.ydata))
            self.dragging = False  # Don't start dragging immediately
        elif event.button == 3:  # Right click
            # Remove AOI rectangle if it exists
            if self.rect is not None:
                self.rect.remove()
                self.rect = None
                self.canvas.draw()

            # Uncheck and trigger the use_aoi_button
            if self.use_aoi_button.isChecked():
                self.use_aoi_button.clicked.emit()
                self.use_aoi_button.setChecked(False)

    def update_temp_rectangle(self):
        """Update the temporary rectangle with debounced drawing."""
        if self.temp_rect is not None and self.current_point is not None:
            w = abs(self.current_point[0] - self.start_point[0])
            h = abs(self.current_point[1] - self.start_point[1])

            # Make square if square AOI button is checked
            if self.square_aoi_button.isChecked():
                size = max(w, h)
                w = h = size

            # Calculate position based on drag direction
            if self.current_point[0] < self.start_point[0]:
                x = self.start_point[0] - w
            else:
                x = self.start_point[0]

            if self.current_point[1] < self.start_point[1]:
                y = self.start_point[1] - h
            else:
                y = self.start_point[1]

            self.temp_rect.set_bounds(x, y, w, h)
            self.canvas.draw()

    def on_mouse_move(self, event):
        """Update rectangle while dragging with debounced drawing."""
        if event.button == 1 and self.start_point:
            # Start dragging if mouse moves while button is held
            if not self.dragging:
                self.dragging = True
                # Initialize temporary rectangle
                if self.rect is not None:
                    self.rect.remove()
                    self.rect = None
                self.temp_rect = plt.Rectangle(
                    self.start_point,
                    0,
                    0,
                    edgecolor="red",
                    facecolor="none",
                    linestyle="--",
                )
                self.ax.add_patch(self.temp_rect)

            # Only update current_point if mouse is inside axes
            if event.inaxes:
                self.current_point = (int(event.xdata), int(event.ydata))
                # Restart the debounce timer
                self.draw_timer.start()

    def on_mouse_release(self, event):
        """Finalize AOI selection on mouse release."""
        if event.button == 3:  # Right click
            return  # Already handled in on_mouse_press

        if self.dragging:
            # Handle drag operation completion
            self.dragging = False

            # Clean up temporary rectangle regardless of where release happened
            if self.temp_rect is not None:
                self.temp_rect.remove()
                self.temp_rect = None
                self.canvas.draw()

            # Only process the selection if release happened inside axes
            if not event.inaxes:
                # Restore previous AOI by triggering redraw
                self.on_aoi_click(None)
                return

            end_point = (int(event.xdata), int(event.ydata))

            # Calculate AOI parameters from drag
            x1, y1 = self.start_point
            x2, y2 = end_point

            center_x = round((x1 + x2) / 2)  # Regular division with rounding
            center_y = round((y1 + y2) / 2)  # Regular division with rounding
            width = max(abs(x2 - x1), 1)  # Ensure minimum width of 1
            height = max(abs(y2 - y1), 1)  # Ensure minimum height of 1

            if self.square_aoi_button.isChecked():
                size = max(width, height)
                width = height = size

            # Block signals to prevent interference from aoi_range_changed
            self.aoi_x_range_input.blockSignals(True)
            self.aoi_y_range_input.blockSignals(True)
            self.aoi_x_pt_input.blockSignals(True)
            self.aoi_y_pt_input.blockSignals(True)

            # Update spinboxes with the actual drawn dimensions
            self.aoi_x_pt_input.setValue(center_x)
            self.aoi_y_pt_input.setValue(center_y)
            self.aoi_x_range_input.setValue(width)
            self.aoi_y_range_input.setValue(height)

            # Update stored values directly
            self.aoi_x_pt = center_x
            self.aoi_y_pt = center_y
            self.aoi_x_range = width
            self.aoi_y_range = height

            self.aoi_x_range_input.blockSignals(False)
            self.aoi_y_range_input.blockSignals(False)
            self.aoi_x_pt_input.blockSignals(False)
            self.aoi_y_pt_input.blockSignals(False)

        else:
            # Handle single click - use current AOI size centered on click point
            self.aoi_x_pt_input.setValue(int(event.xdata))
            self.aoi_y_pt_input.setValue(int(event.ydata))

        # Draw final rectangle and update AOI
        self.on_aoi_click(None)
        self.canvas.draw()

        # Reset state
        self.start_point = None
        self.current_point = None

    def on_square_button_clicked(self):
        """Handle square AOI button click by making current AOI square."""
        if self.square_aoi_button.isChecked():
            # Get current dimensions
            width = self.aoi_x_range
            height = self.aoi_y_range

            # Make square using larger dimension
            size = max(width, height)

            # Update the spinboxes which will trigger redraw
            self.aoi_x_range_input.setValue(size)
            self.aoi_y_range_input.setValue(size)
