from __future__ import annotations

import asyncio
import os
import platform
from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pylablib as pll
import pylablib.devices.Andor
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qscope.device.device import Device
from qscope.util.logging import format_error_response

if TYPE_CHECKING:
    from qscope.system import System
    from qscope.types import ServerConnection

this_folder = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
lib_folder = os.path.abspath(
    os.path.join(this_folder, *[".." for i in range(4)], "proprietary_artefacts")
)

# TODO could probably have a class above this for all pylablib cameras, share functionality
class AndorSDK3(Device):
    def __init__(self, **config_kwargs):
        super().__init__(**config_kwargs)

        if platform.system() == "Windows":
            # pll.par["devices/dlls/andor_sdk3"] = lib_folder
            pll.par["devices/dlls/andor_sdk3"] = "C:\\Qscope\\proprietary_artefacts"
        else:
            pll.par["devices/only_windows_dlls"] = False
            pll.par["devices/dlls/andor_sdk3"] = "/usr/local/lib/libatcore.so"

        self._binning = (1, 1)
        self._exposure_time = 0.3

        self.video_stop_event = asyncio.Event()

    def open(self) -> tuple[bool, str]:
        try:
            self.cam = pylablib.devices.Andor.AndorSDK3Camera()
            self.cam.open()
            if pylablib.devices.Andor.get_cameras_number_SDK3() == 2 and not isinstance(
                self, AndorSimCam
            ):
                # only simcams, but we don't want a simcam (first two cams are always simcams)
                self.cam.close()
                raise RuntimeError(
                    f"Can't find Camera of type {self.__class__.__name__}"
                )

            self._set_image_indexing("xyb")
            # self._set_frame_format("chunks")
            self._set_frame_format("array")
            self._set_attr_dev("SensorCooling", True)  # is this ok for all cams?
            self._set_attr_dev("SimplePreAmpGainControl", 2)
            self.cam.name = self._get_attr_dev("CameraModel")

            self.set_shutter_mode(self._shutter_mode)
            self.set_roi(self._roi)
            self.set_hardware_binning(self._binning)
            self.set_exposure_time(self._exposure_time)
            # self._sensor_width = self._get_attr_dev("SensorWidth")
            # self._sensor_height = self._get_attr_dev("SensorHeight")
            self.set_trigger_mode(
                "ext_exp"
            )  # use b. deflt, will switch to int when needed
            # Set up the camera gain
            logger.info(f"Connected to Camera: {self.cam.get_device_info()}")
            return True, f"Connected to Camera: {self.cam.get_device_info()}"
        except Exception as e:
            logger.exception("Error connecting to Camera.")
            return False, f"Error connecting to Camera: {format_error_response()}"

    def close(self):
        if hasattr(self, "cam"):
            self.cam.close()
            logger.info(
                "Disconnected from Camera: {}",
                self.cam.name if hasattr(self.cam, "name") else "Unknown camera",
            )

    def is_connected(self):
        if not hasattr(self, "cam"):
            return False
        return self.cam.is_opened()

    def get_readout_time(self):
        # returns the readout time in seconds
        readout_time = self._get_attr_dev("ReadoutTime")
        # add 10% more time to be safe
        return readout_time + 0.2 * readout_time

    def take_snapshot(self):
        self.set_trigger_mode("int")
        image = self.cam.snap()
        self.set_trigger_mode("ext_exp")
        return image

    def get_all_seq_frames(self):
        # TODO what shape is returned? Need some more docs around here.
        return self.cam.read_multiple_images()

    def setup_acquisition(self, mode="sequence", nframes=1):
        self.cam.setup_acquisition(mode="sequence", nframes=nframes)

    def clear_acquisition(self):
        self.cam.clear_acquisition()

    def start_acquisition(self):
        self.cam.start_acquisition()

    def stop_acquisition(self):
        self.cam.stop_acquisition()

    def get_trigger_time(self) -> float:
        return self.get_readout_time()

    # def get_amp_mode(self, *args, **kwargs):
    #     return self._cam.get_amp_mode(full=True)

    #################################################################
    # set camera parameters                                         #

    def set_trigger_mode(self, mode):
        self.cam.set_trigger_mode(mode)

    def set_shutter_mode(self, mode):
        # sets the shutter mode to either "Rolling" or "Global"
        # also defines the trigger mode to the correct format for each mode
        if mode == "Rolling":
            self._set_attr_dev("ElectronicShutteringMode", "Rolling")
            self.set_trigger_mode("ext_exp")
        elif mode == "Global":
            self._set_attr_dev("ElectronicShutteringMode", "Global")
            self.set_trigger_mode("ext")
        else:
            logger.error("Shutter mode {} not recognised.", mode)

    def get_shutter_mode(self):
        self._shutter_mode = self._get_attr_dev("ElectronicShutteringMode")
        return self._shutter_mode

    def _set_image_indexing(self, indexing):
        self.cam.set_image_indexing(indexing)

    def _set_frame_format(self, fmt):
        """
        Can be "list" (list of 2D arrays),
        "array" (a single 3D array),
        "chunks" (list of 3D “chunk” arrays;
        """
        # TODO we want to set this to 'chunks' for measurements/sequences
        # (or 'try_chunks')?
        self.cam.set_frame_format(fmt)

    def _get_attr_dev(self, attr):
        return self.cam.get_attribute_value(attr)

    def _set_attr_dev(self, attr, value):
        # the attributes for Zyla are case sensitive so we need to make sure the
        # attribute is in the correct case

        # cross check the attribute with the list of attributes and adjust the case
        # if necessary
        attr_list = self.cam.get_all_attributes()
        # if the attribute matches one in the list when it is all lower case then
        # set the attribute to the one in the list
        new_attr = [attr_ for attr_ in attr_list if attr.lower() == attr_.lower()][0]
        self.cam.set_attribute_value(new_attr, value)

    #################################################################
    #               CAMERA PROPERTIES                               #
    #################################################################

    def get_roi(self):
        # this returns both the roi and the binning to keep them seperate for
        # easier data handling we define both seperately
        roi_and_bin = self.cam.get_roi()
        self._roi = roi_and_bin[0:4]
        return self._roi

    def set_roi(self, roi):
        if len(roi) != 4:
            # then the user has passed a centered roi with a radius
            roi = [
                int(self._sensor_width / 2 - roi[0] / 2),
                int(self._sensor_width / 2 + roi[0] / 2),
                int(self._sensor_height / 2 - roi[1] / 2),
                int(self._sensor_height / 2 + roi[1] / 2),
            ]
        # TODO add check for roi size within sensor size?
        self.cam.set_roi(*roi, *self.get_hardware_binning())
        self._roi = roi
        self.update_data_size()
        return self._roi

    def get_hardware_binning(self):
        roi_and_bin = self.cam.get_roi()
        self._binning = roi_and_bin[4:6]
        return self._binning

    def set_hardware_binning(self, binning):
        roi = self.get_roi()
        self.cam.set_roi(*roi, *binning)
        self._frame_shape = [roi[1] - roi[0], roi[3] - roi[2]]
        self._binning = binning
        self.update_data_size()

    def get_frame_shape(self):
        roi = self.get_roi()
        self._frame_shape = [roi[1] - roi[0], roi[3] - roi[2]]
        return self._frame_shape

    def set_frame_shape(self, frame_shape):
        roi = [
            self._sensor_width / 2 - frame_shape[0] / 2,
            self._sensor_width / 2 + frame_shape[0] / 2,
            self._sensor_height / 2 - frame_shape[1] / 2,
            self._sensor_height / 2 + frame_shape[1] / 2,
        ]
        # roi = [int(i) for i in roi] # need one of these?
        self.cam.set_roi(*roi, *self.get_hardware_binning())

        self.set_roi(roi)
        logger.info("Set camera roi to: {}", roi)
        self._frame_shape = frame_shape
        self.update_data_size()
        return self._frame_shape

    def update_data_size(self):
        self._data_size = self.cam.get_data_dimensions()
        return self._data_size

    def get_data_size(self):
        return self._data_size

    def get_exposure_time(self):
        self._exposure_time = self.cam.cav["ExposureTime"]
        logger.info("Exposure time: {}", self._exposure_time)
        return self._exposure_time

    def set_exposure_time(self, value):
        # to set the exposure time we need to make sure the trigger mode is set to
        # internal
        trig_mode = self._get_attr_dev("TriggerMode")
        self.cam.set_trigger_mode("int")
        self.cam.set_exposure(value)
        # switch back to the original trigger mode
        if trig_mode == "Internal":
            self.cam.set_trigger_mode("int")
        elif trig_mode == "External Exposure":
            self.cam.set_trigger_mode("ext_exp")
        self._exposure_time = value

    def wait_for_frame(self):
        return self.cam.wait_for_frame(error_on_stopped=True)

    async def start_video(self, connection):
        logger.info("Starting video feed")

        async def video_loop(cn: ServerConnection, stop_event: asyncio.Event):
            cn.reset_stream()
            # start the camera acquisition
            self.cam.set_trigger_mode("int")
            self.cam.setup_acquisition(mode="sequence", nframes=15)
            self.cam.start_acquisition()
            await asyncio.sleep(0)
            while True:
                await asyncio.sleep(0)
                try:
                    if stop_event.is_set():
                        stop_event.clear()
                        logger.info("Stopping video feed")
                        self.cam.stop_acquisition()
                        # set the trigger mode back to external exposure
                        self.set_trigger_mode("ext_exp")
                        break
                    self.cam.wait_for_frame()
                    ar = np.array(self.get_all_seq_frames())[0, ::]
                    cn.send_stream_chunk(ar, header="video")
                except Exception as e:
                    logger.exception("Error in video loop.")
                    break

        self.video_stop_event.clear()
        self.task = asyncio.create_task(video_loop(connection, self.video_stop_event))
        await asyncio.sleep(0)  # let above start

    def stop_video(self):
        self.video_stop_event.set()

    def start_popup_video(self, exposure=None, blit=False):
        """
        This is to test the circular buffer mode of the camera
        """
        mpl.use("Qt5Agg")
        if exposure is not None:
            self.exposure = exposure

        frame = self.take_snapshot()

        # determine the order of magnitude for the summed data
        sum_pl = np.sum(frame, axis=(0, 1))

        # start the camera acquisition
        self.cam.set_trigger_mode("int")
        self.cam.setup_acquisition(mode="sequence", nframes=15)
        self.cam.start_acquisition()

        plt.ion()
        fig = plt.figure(figsize=(13, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

        # add a buffer to the range for count changes
        cmax = np.max(frame) + 0.1 * np.max(frame)
        cmin = np.min(frame) - 0.1 * np.min(frame)

        img = ax1.imshow(
            frame,
            interpolation="None",
            cmap="gray",
            vmin=cmin,
            vmax=cmax,
            origin="upper",
        )
        # Add a colorbar to the image
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(img, cax=cax, orientation="vertical")

        # Create the time trace
        sum_pl = np.sum(frame, axis=(0, 1))
        time = 0
        (line,) = ax2.plot(time, sum_pl)
        # axes labels
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Intensity (cps)")
        ax2.set_xlim(0, 5)

        # add a space between the plots
        fig.subplots_adjust(wspace=0.3)

        fig.canvas.draw()  # note that the first draw comes before setting data

        # cache the background
        if blit:
            axbackground = fig.canvas.copy_from_bbox(ax1.bbox)
            ax2background = fig.canvas.copy_from_bbox(ax2.bbox)

        plt.show(block=False)

        b_draw = False
        n_frames = 0
        sum_pl = np.array([])
        while True:
            # Get a frame from the camera
            self.cam.wait_for_frame()
            frames = self.get_all_seq_frames()
            n_frames += frames.shape[0]

            # sum each frame to get the total intensity
            sum_pl = np.append(sum_pl, np.sum(frames, axis=(1, 2)))
            time = self._exposure_time * np.arange(sum_pl.shape[0])

            # set the new data
            cmax = np.max(frames[-1, ::])
            cmin = np.min(frames[-1, ::])

            img.set_data(frames[-1, ::])
            img.set_clim(cmin, cmax)

            line.set_data(time, sum_pl)

            if blit:
                # restore background
                fig.canvas.restore_region(axbackground)
                fig.canvas.restore_region(ax2background)

                # redraw just the points
                ax1.draw_artist(img)
                ax2.draw_artist(line)

                # update the axes
                ax2.set_ylim(np.min(sum_pl), np.max(sum_pl))
                # check if x axis needs to be updated
                if time[-1] > ax2.get_xlim()[1]:
                    ax2.set_xlim(time[0], time[-1] + 5)
                    b_draw = True

                # fill in the axes rectangle
                fig.canvas.blit(ax1.bbox)
                fig.canvas.blit(ax2.bbox)
            else:
                ax2.set_ylim(np.min(sum_pl), np.max(sum_pl))

                if time[-1] > ax2.get_xlim()[1]:
                    ax2.set_xlim(time[0], time[-1] + 5)

                b_draw = True

            if b_draw:
                fig.canvas.draw()
                b_draw = False

            fig.canvas.flush_events()

            # stop if the user closes the figure
            if not plt.fignum_exists(fig.number):
                break

        # stop the camera acquisition
        self.cam.stop_acquisition()
        # set the trigger mode back to external exposure
        self.set_trigger_mode("ext_exp")

        # close the figure
        plt.close(fig)

        return time, sum_pl


class AndorSimCam(AndorSDK3):
    pass


class Sona42(AndorSDK3):
    def __init__(self, **config_kwargs):
        super().__init__(**config_kwargs)
        self._shutter_mode = "Rolling"
        self._roi = (0, 2048, 0, 2048)  # correct?

    def open(self) -> tuple[bool, str]:
        # FIXME, SONA temp is set differently.
        super().open()
        self.cam.set_temperature(-25, enable_cooler=True)
        self._set_attr_dev("SensorCooling", True)


class Zyla42(AndorSDK3):
    def __init__(self, **config_kwargs):
        super().__init__(**config_kwargs)
        self._shutter_mode = (
            "Rolling"  # New Zyla 42 is rolling only. (use rolling by default anyway?)
        )
        self._roi = (0, 2048, 0, 2048)
        self._sensor_height = 2048
        self._sensor_width = 2048


class Zyla55(AndorSDK3):
    def __init__(self, **config_kwargs):
        super().__init__(**config_kwargs)
        self._shutter_mode = "Rolling"
        # is this the correct order(ing)?
        self._roi = (0, 2056, 0, 2048)
        self._sensor_height = 2056
        self._sensor_width = 2048

    def open(self) -> tuple[bool, str]:
        is_ok, reason = super().open()
        if is_ok:
            try:
                self.cam.set_temperature(-20, enable_cooler=True)
                self._set_attr_dev("SensorCooling", True)
            except Exception as e:
                return False, f"Error setting temp: {e}"
        return is_ok, reason
