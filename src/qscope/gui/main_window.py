# Python script for creating the main window of the QDM GUI


import asyncio
import os
import pickle
import threading
import time
from typing import Optional

import numpy as np
import zmq
from loguru import logger
from PyQt6 import QtCore
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *

# Menu imports
import qscope.gui.menu
import qscope.server

# Load the other UI elements
from qscope.gui.camera.camera_window import init_video_tab
from qscope.gui.qmeas.qmeas_window import init_measurement_tab
from qscope.gui.util import set_matplotlib_style, show_warning
from qscope.meas import MEAS_STATE
from qscope.server.connection_manager import ConnectionManager
from qscope.types import (
    MeasurementFrame,
    MeasurementUpdate,
    NewMeasurement,
    Notification,
    RollingAvgSweepUpdate,
    SaveFullComplete,
    SweepUpdate,
)
from qscope.util import format_error_response


class MainWindow(QMainWindow):
    # Create a signal to emit data from the listener thread
    # new_data_signal = QtCore.pyqtSignal(dict, name="new_data_signal")
    notif_signal = QtCore.pyqtSignal(Notification, name="new_meas_signal")
    new_video_signal = QtCore.pyqtSignal(list, name="new_video_signal")

    connection_manager: ConnectionManager

    rolling_meas_window: int = 1
    rolling_meas_idxs: list[int]
    rolling_meas_sig_array: np.ndarray[float]
    rolling_meas_ref_array: np.ndarray[float]

    def __init__(
        self,
        system_name: Optional[str] = None,
        auto_connect: bool = True,
        host: str = "",
        msg_port: str = "",
    ):
        super().__init__()

        # Flag to track frame processing status
        self.processing_frame = False

        # Make a window object
        self.setWindowTitle("QScope GUI")
        self.setGeometry(0, 0, 1700, 950)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "icon.png")
        self.setWindowIcon(QIcon(icon_path))

        # detect if the OS is windows and set the style to dark
        scheme = QGuiApplication.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Dark:
            set_matplotlib_style("dark")
        else:
            set_matplotlib_style("light")

        self.close_signal = False

        # create the menu bar
        self._create_menu_bar()

        # Add a tab widget with tabs for different settings in the GUI
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # set the central widget so that the tabs are the main part of the window
        self.tabs.setTabPosition(QTabWidget.TabPosition.West)

        # Add the tabs to the tab widget
        self.tab1, self.tab2, self.tab3, self.tab4 = (
            QWidget(self.tabs),
            QWidget(self.tabs),
            QWidget(self.tabs),
            QWidget(self.tabs),
        )

        self.tabs.addTab(self.tab1, "Video")
        self.tabs.addTab(self.tab2, "Measurements")
        self.tabs.addTab(self.tab3, "Electrical")
        self.tabs.addTab(self.tab4, "Temperature")

        # set the font to be bold
        self.tabs.tabBar().setStyleSheet("font-weight: bold; font-size: 12pt;")

        # self.tabs.tabBar().setFont(QFont("sans-serif", 12))

        # Initialize data storage variables
        self.frame = np.random.rand(512, 512)
        self.roi_img = np.random.rand(10, 10)
        self.x_pixel = 0
        self.y_pixel = 0
        self.roi_x_range = 10
        self.roi_y_range = 10

        # Initialize the tabs
        init_video_tab(self, self.tab1)
        init_measurement_tab(self, self.tab2)

        # Create the quit action
        quit_action_1 = QAction("Quit", self)
        quit_action_1.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action_1.triggered.connect(self.closedown)
        self.addAction(quit_action_1)
        quit_action_2 = QAction("Close", self)
        quit_action_2.setShortcut(QKeySequence.StandardKey.Close)
        quit_action_2.triggered.connect(self.closedown)
        self.addAction(quit_action_2)

        # Create the status bar
        self.statusBar = qscope.gui.menu.QScopeStatusBar(self)
        self.setStatusBar(self.statusBar)

        self.connection_manager = ConnectionManager()
        self._connect(system_name, auto_connect, host, msg_port)

        if self.connection_manager.is_connected():
            self.start_notif_listener_thread()
            self.start_video_listener_thread()

        self.clock = QElapsedTimer()
        self.clock.start()

        self.clock_meas = QElapsedTimer()
        self.clock_meas.start()

        if self.connection_manager.is_connected():
            self.statusBar.set_server_status("Connected")
            self.statusBar.set_hw_status("Initialized")
            self.server_menu._set_connection_status(status=True)
            
            try:
                self.statusBar.set_hw_status("Initialized" if self.parent.connection_manager.client_sync.hardware_started_up else "Not initialized")
            except:
                pass # what to do?
        else:
            self.statusBar.set_server_status("Not connected")
            self.statusBar.set_hw_status("Not initialized")
            self.server_menu._set_connection_status(status=False)

    def _connect(
        self,
        system_name: Optional[str],
        auto_connect: bool = True,
        host: str = "",
        msg_port: str = "",
    ):
        if auto_connect:
            logger.info("Attempting to find local running server.")
            try:
                # try local server QUICKLY
                self.connection_manager.connect(timeout=0.1, request_retries=1)
                if system_name:  # check it matches request sys etc.
                    client_sync = self.connection_manager.client_sync
                    if client_sync.system_name.lower() != system_name.lower():
                        show_warning(
                            "Local running system is wrong system: "
                            + f"{client_sync.system_name.lower()} vs"
                            + f"{system_name.lower()}.\n"
                            + f"Continuing with local running system."
                        )
                if self.connection_manager.client_sync.version != qscope.__version__:
                    show_warning(
                        "Server and client versions are different: "
                        + f"{self.connection_manager.client_sync.version} vs "
                        f"{qscope.__version__}"
                    )
                if self.connection_manager.is_connected():
                    self.connection_manager.startup()  # will do nothing if already started up
                    self.connection_manager.ping(
                        request_retries=3
                    )  # test that server responds
                logger.info("Connected to local running server.")
                return
            except:
                logger.info("Failed to find local running server.")
                pass  # meh
        if system_name:
            logger.info("Attempting to start local server.")
            try:
                self.connection_manager.start_local_server(
                    system_name=system_name,
                    log_to_file=True,
                    log_level="TRACE", # TODO use setting passed from cli
                )
                self.connection_manager.connect()
                if self.connection_manager.client_sync.version != qscope.__version__:
                    show_warning(
                        "Server and client versions are different: "
                        + f"{self.connection_manager.client_sync.version} vs "
                        f"{qscope.__version__}"
                    )
                self.connection_manager.startup()
                self.connection_manager.ping(
                    request_retries=3,
                )  # test that server responds

                logger.info("Connected to background server.")
            except:
                raise RuntimeError(
                    "Error starting bg server: {}", format_error_response()
                )

    def closedown(self):
        """Close the window and clean up resources"""
        event = QCloseEvent()
        self.closeEvent(event)
        self.close()

    def closeEvent(self, event):
        """Handle the window close event to clean up resources."""
        self.close_signal = True

        # Give threads a moment to notice the close signal
        time.sleep(0.2)

        # Close connection manager if it exists
        if hasattr(self, "connection_manager"):
            # TODO when stable swap these two. (user may want to connect to running instance)
            # (will need buttons to disconnect/connect then)
            # self.connection_manager.stop_server()
            self.connection_manager.disconnect()

    def handle_save(self):
        """Handle Save action based on current tab"""
        current_tab = self.tabs.currentWidget()

        try:
            if current_tab == self.tab2:  # Measurements tab
                if hasattr(self, "meas_id"):
                    # Simulate clicking the save measurement button
                    self.meas_opts.save_measurement_button.click()
                else:
                    show_warning("No measurement data to save")
            elif current_tab == self.tab1:  # Video tab
                # Simulate clicking the camera save button
                self.cam_opts.save_image_button.click()
        except Exception as e:
            show_warning(f"Error during save operation: {str(e)}")

    def handle_save_as(self):
        """Handle Save As action based on current tab"""
        current_tab = self.tabs.currentWidget()

        try:
            if current_tab == self.tab2:  # Measurements tab
                if hasattr(self, "meas_id"):
                    # Simulate clicking the (qmeas) save image button
                    self.meas_opts.save_image_button.click()
                else:
                    show_warning("No measurement data to save")
        except Exception as e:
            show_warning(f"Error during save-as operation: {str(e)}")

    # Make a menu bar
    def _create_menu_bar(self):
        menuBar = QMenuBar(self)
        self.setMenuBar(menuBar)
        menuBar.setNativeMenuBar(False)

        # Add keyboard shortcuts without menu items
        saveAction = QAction("Save", self)
        saveAction.setShortcut(QKeySequence.StandardKey.Save)
        saveAction.triggered.connect(self.handle_save)
        self.addAction(saveAction)

        saveAsAction = QAction("Save All", self)
        saveAsAction.setShortcut(QKeySequence.StandardKey.SaveAs)
        saveAsAction.triggered.connect(self.handle_save_as)
        self.addAction(saveAsAction)

        # Settings menu
        self.settings_menu = qscope.gui.menu.SettingsMenu(self, menuBar)

        # Server menu
        self.server_menu = qscope.gui.menu.ServerMenu(self, menuBar)

        # Notes menu
        self.notes_menu = qscope.gui.menu.NotesMenu(self, menuBar)

        # Help menu
        self.help_menu = qscope.gui.menu.HelpMenu(self, menuBar)

    ### Listener Threads ###

    # Listener for the measurement tab
    def notif_socket_thread(self):
        """Start the listener thread and connect signal to slot"""
        old_data = None

        while not self.close_signal:
            try:
                socket = self.connection_manager.get_notification_socket()
                msg = socket.recv()
                notif = Notification.from_msgpack(msg)
                self.notif_signal.emit(notif)
                asyncio.run(asyncio.sleep(0.01))
            except zmq.error.ContextTerminated:
                # Context was terminated, exit gracefully
                logger.info("ZMQ context terminated, stopping notification thread")
                break
            except zmq.error.ZMQError as e:
                # Handle other ZMQ errors
                logger.debug(f"ZMQ error in notification thread: {e}")
                # Small sleep to prevent tight loop if errors persist
                time.sleep(0.1)
            except Exception as e:
                logger.exception(f"Unexpected error in notification thread: {e}")
                # Small sleep to prevent tight loop if errors persist
                time.sleep(0.1)

    def start_notif_listener_thread(self):
        """Start the listener thread and connect signal to slot"""

        self.listener_thread = threading.Thread(target=self.notif_socket_thread)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        self.notif_signal.connect(self.handle_new_notif)

    def handle_new_notif(self, notif):
        """Update the measurement data"""

        match notif.type:
            case NewMeasurement.type:
                # set the measurement ID
                self.meas_id = notif.meas_id
                self.meas_opts.meas_id_indicator.setText(self.meas_id)

            case MeasurementUpdate.type:
                # Update the measurement state

                self.meas_id = notif.meas_id
                self.meas_opts.meas_id_indicator.setText(self.meas_id)

                self.update_meas_state(notif.new_state)

            case RollingAvgSweepUpdate.type:
                self.rolling_meas_idxs = notif.sweep_idxs
                self.rolling_meas_sig_array = notif.sweep_ysig
                self.rolling_meas_ref_array = notif.sweep_yref

                # TODO we need a util normalisation function
                # Use the normalisation for the line plot
                match self.line_figure_opts.plot_norm_dropdown.currentText():
                    case "None":
                        data_frame = self.rolling_meas_sig_array
                    case "Subtract":
                        data_frame = (
                            self.rolling_meas_sig_array - self.rolling_meas_ref_array
                        )
                    case "Divide":
                        data_frame = (
                            self.rolling_meas_sig_array / self.rolling_meas_ref_array
                        )
                    case "Normalise":
                        data_frame = 100 * (
                            1
                            - self.rolling_meas_sig_array / self.rolling_meas_ref_array
                        )
                    case _:
                        data_frame = self.rolling_meas_sig_array

                self.rolling_img_figure.update_data(
                    data_frame, self.rolling_meas_idxs, notif.sweep_x
                )

                if self.line_figure_opts.real_time_button.isChecked():
                    # update the real-time plot
                    self.meas_line_figure.set_data(
                        notif.sweep_x,
                        notif.rolling_meas_sig_array[-1, :],
                        notif.rolling_meas_ref_array[-1, :],
                    )

            case SweepUpdate.type:
                elapsed_time = self.clock_meas.elapsed()

                # Make the realtime data from the averaged data
                if notif.nsweeps == 0:
                    # first sweep of measurements, no previous data
                    self.previous_sweep_data = notif.sweep_data.copy()
                    self.previous_sweep_data[1] = self.previous_sweep_data[1] * 0
                    self.previous_sweep_data[2] = self.previous_sweep_data[2] * 0

                if notif.sweep_progress == 0.0 and notif.nsweeps > 1:
                    # Update the previous sweep data if the sweep is finished and convert it to a sum of the data
                    self.previous_sweep_data = notif.sweep_data * notif.nsweeps

                # Define the realtime data
                self.realtime_x = notif.sweep_data[0]
                self.realtime_y = (notif.nsweeps + 1) * notif.sweep_data[
                    1
                ] - self.previous_sweep_data[1]
                self.realtime_y_ref = (notif.nsweeps + 1) * notif.sweep_data[
                    2
                ] - self.previous_sweep_data[2]

                # Remove the elements that are not in the current sweep
                indices = len(self.realtime_x) * notif.sweep_progress / 100
                self.realtime_x = self.realtime_x[: int(indices)]
                self.realtime_y = self.realtime_y[: int(indices)]
                self.realtime_y_ref = self.realtime_y_ref[: int(indices)]

                if elapsed_time > 50:  # 50ms refresh rate.
                    self.meas_id = notif.meas_id
                    self.meas_opts.meas_id_indicator.setText(self.meas_id)
                    # Update the number of sweeps
                    self.meas_opts.ith_loop.setValue(notif.nsweeps)

                    # check if the sweep full sweep has finished and send force fit if true
                    if notif.sweep_progress == 0.0:
                        force_fit = True
                    else:
                        force_fit = False

                    # update the sweep percentage
                    self.meas_opts.sweep_progress.setValue(int(notif.sweep_progress))

                    # update the realtime data
                    self.meas_line_figure.update_realtime_data(
                        self.realtime_x,
                        self.realtime_y,
                        self.realtime_y_ref,
                    )

                    # update the sweep data
                    self.meas_line_figure.set_data(
                        notif.sweep_data[0],
                        notif.sweep_data[1],
                        notif.sweep_data[2],
                        force_fit=force_fit,
                    )

                    self.clock_meas.restart()

                    if (
                        notif.nsweeps >= self.meas_opts.stop_after_sweeps_idx.value()
                        and self.meas_opts.stop_after_sweeps.isChecked()
                    ):
                        # tell the server to stop the measurement at the end of the sweep
                        qscope.server.stop_measurement(self.client, notif.meas_id)

                    if (
                        notif.nsweeps >= self.meas_opts.save_after_sweeps_idx.value()
                        and self.meas_opts.save_after_sweeps.isChecked()
                    ):
                        # Check if the number of sweeps is an integer multiple of the save after sweeps idx
                        if (
                            notif.nsweeps % self.meas_opts.save_after_sweeps_idx.value()
                            == 0
                        ):
                            # save the measurement data
                            qscope.server.measurement_save_sweep(
                                self.client, notif.meas_id, self.get_project_name()
                            )

            case MeasurementFrame.type:
                self.meas_id = notif.meas_id
                self.meas_opts.meas_id_indicator.setText(self.meas_id)
                # Update the measurement frame (sig_frame, ref_frame)
                self.frame_figure.update_data(notif.sig_frame, notif.ref_frame)

            case SaveFullComplete.type:
                # just update the button
                self.meas_opts.save_image_button.setStyleSheet("background-color: None")
                self.meas_opts.save_image_button.setChecked(False)

    def update_meas_state(self, state):
        """Update the measurement state"""
        self.meas_opts.state_indicator.setText(state)
        if state == MEAS_STATE.RUNNING:
            # Change the background color of the state label to green
            self.meas_opts.state_indicator.setStyleSheet("background-color: green")
        elif (
            state == MEAS_STATE.PAUSED
            or state == MEAS_STATE.PREPARING
            or state == MEAS_STATE.AWAITING_START
        ):
            self.meas_opts.state_indicator.setStyleSheet("background-color: red")
        else:
            self.meas_opts.state_indicator.setStyleSheet("background-color: None")

    # Listener for the video tab
    def video_socket_thread(self):
        """Start the listener thread and connect signal to slot"""
        old_data = None
        frame_buffer = []
        MAX_BUFFER_SIZE = 10  # Maximum number of frames to buffer

        try:  # confirm connection first
            self.connection_manager.ping(request_retries=1)
        except Exception as e:
            logger.exception(f"Unexpected error in starting video thread: {e}")
            return  # ? acceptable ?

        while not self.close_signal:
            try:
                if self.connection_manager._connection is None:
                    continue
                socket = self.connection_manager.get_stream_socket()

                # Check if we're falling behind (buffer getting too full)
                if len(frame_buffer) > MAX_BUFFER_SIZE:
                    # Drop oldest frames, keeping only the most recent ones
                    frames_to_keep = MAX_BUFFER_SIZE // 2
                    frame_buffer = frame_buffer[-frames_to_keep:]
                    logger.debug(
                        f"Client falling behind, dropped {MAX_BUFFER_SIZE - frames_to_keep} frames"
                    )

                # Non-blocking receive with timeout
                try:
                    new_data = socket.recv_multipart(flags=zmq.NOBLOCK)
                    frame_buffer.append(new_data)
                except zmq.Again:
                    # No data available, sleep a bit
                    time.sleep(0.01)

                # If we have frames and we're not currently processing one, emit the newest
                if frame_buffer and not self.processing_frame:
                    self.processing_frame = True
                    self.new_video_signal.emit(frame_buffer.pop(0))

            except zmq.error.ContextTerminated:
                # Context was terminated, exit gracefully
                logger.info("ZMQ context terminated, stopping video thread")
                break
            except zmq.error.ZMQError as e:
                # Handle other ZMQ errors
                logger.debug(f"ZMQ error in video thread: {e}")
                time.sleep(0.1)
            except Exception as e:
                logger.exception(f"Unexpected error in video thread: {e}")
                time.sleep(0.1)

    def start_video_listener_thread(self):
        """Start the listener thread and connect signal to slot"""
        import threading

        # Use lists instead of numpy arrays for better append performance
        self.timetrace_data = []
        self.timetrace_time = []

        self.listener_thread = threading.Thread(target=self.video_socket_thread)
        self.listener_thread.daemon = True
        self.listener_thread.start()
        self.new_video_signal.connect(self.handle_new_video_frame)

    def handle_new_video_frame(self, message):
        """Update the video"""
        try:
            header, frame = message
            assert header == b"video"
            frame = pickle.loads(frame)
            # frame = np.frombuffer(frame, dtype=np.uint16).reshape(2048, 2048) # FIXME

            # track the sum of the frames for the mpl
            # Use list append instead of np.append for better performance
            self.timetrace_data.append(np.sum(frame))
            self.timetrace_time.append(
                len(self.timetrace_data) * self.cam_opts.exposure_time_input.value()
            )

            # # Downsample when we exceed the maximum points
            # MAX_POINTS = 1000  # Target maximum number of points
            # if len(self.timetrace_data) > MAX_POINTS * 1.2:  # Add some hysteresis
            #     # Calculate the decimation factor needed
            #     decimation_factor = len(self.timetrace_data) // (MAX_POINTS // 2)

            #     # Use every Nth point to maintain even time steps
            #     self.timetrace_data = self.timetrace_data[::decimation_factor]
            #     self.timetrace_time = self.timetrace_time[::decimation_factor]

            #     logger.debug(
            #         f"Downsampled time trace data with factor {decimation_factor}, new size: {len(self.timetrace_data)}"
            #     )
            if len(self.timetrace_data) > 50000:
                logger.warning("Timetrace reached 50k pts, consider adding downsampling in `handle_new_video_frame`.")

            elapsed_time = self.clock.elapsed()

            if elapsed_time > 50:  # 50ms refresh rate.
                # First update the timetrace figure (faster operation)
                # Convert to numpy arrays only when needed for plotting
                self.video_timetrace_fig.fast_update(
                    np.array(self.timetrace_time), np.array(self.timetrace_data)
                )

                # Then update the video figure (slower operation)
                self.video_fig.update_data(frame)

                self.clock.restart()
        finally:
            # Signal that we're done processing this frame
            self.processing_frame = False

    def get_project_name(self) -> str:
        return self.statusBar.get_project_tag()

    def get_notes(self) -> str:
        return self.notes_menu.get_notes()

    def show_status_msg(self, msg: str, msecs=10000):
        self.statusBar.showMessage(msg, msecs)
