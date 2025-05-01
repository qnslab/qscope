from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
import os
import numpy as np
import numpy.random
from PIL import Image
from loguru import logger

from qscope.device.device import Device
from qscope.types.protocols import CameraProtocol

if TYPE_CHECKING:
    from qscope.types import ServerConnection

this_folder = os.path.abspath(os.path.realpath(os.path.dirname(__file__)))
frame_logo_path = os.path.abspath(
    os.path.join(this_folder, *[".." for i in range(4)], "docs/images/frame_logo.png")
)


class MockCamera(Device):  # Protocol compliance checked by role system
    def __init__(self, **config_kwargs):
        super().__init__(**config_kwargs)
        self._roi = (0, 2560, 0, 2560)
        self._binning = (1, 1)
        self._exposure_time = 0.1
        self._shutter_mode = "Global"
        self._nframes = 1

        self._image_indexing = "xyb"
        self._frame_format = "array"
        self.SensorWidth = 2560
        self.SensorHeight = 2560

        # use with self._rng.random(shape) for nums in [0.0, 1.0)
        self.__rng = numpy.random.default_rng()

        self.video_stop_event = asyncio.Event()

    # NOTE many of these will be empty/passing, no comms/setup required for mock.

    # def unroll_metadata(self):

    def is_connected(self) -> bool:
        return True

    def get_trigger_time(self):
        return 0.1  # why not

    def open(self) -> tuple[bool, str]:
        logger.info("Connected to Camera: MockCamera")
        return True, "Connected to Camera: MockCamera"

    def close(self):
        logger.info("Disconnected from Camera: {}", "MockCamera")

    def get_readout_time(self):
        return 0.1 * self.get_exposure_time()

    def take_snapshot(self):
        shp = self.get_frame_shape()

        # Load the PNG file
        img = Image.open(frame_logo_path).convert("L")  # Convert to grayscale

        # Convert to binary (0 or 1) using a threshold
        threshold = 128
        binary_img = img.point(lambda p: 200 if p < threshold else 100)

        # Resize the image to match the frame shape
        resized_img = binary_img.resize((shp[1], shp[0]), Image.NEAREST)

        # Convert the image to a NumPy array
        frame = np.array(resized_img, dtype=np.uint8)


        frame_noise = self.__rng.integers(
            low=0,
            high=np.random.randint(1, high=80),
            size=self.get_frame_shape(),
            dtype=np.uint8,
        )

        # Add the noise to just the background of the image
        # get all the pixels that are not part of the logo
        # and set them to 0 in the noise image

        mask = np.where(frame == 200, 0, 1)  # Create a mask for the logo pixels
        frame_noise = np.where(mask == 0, 0, frame_noise)
        # set the logo pixels to 0 in the noise image


        return frame + frame_noise

    # unsure if this is the correct data shape.
    def get_all_seq_frames(self):
        return self.__rng.random((self._nframes, *self.get_frame_shape()))

    def setup_acquisition(self, mode="sequence", nframes=1):
        self._nframes = nframes

    def start_acquisition(self):
        pass

    def stop_acquisition(self):
        pass

    def clear_acquisition(self):
        pass

    def wait_for_frame(self):
        pass

    #################################################################
    # set camera parameters                                         #

    def set_trigger_mode(self, mode):
        pass

    def set_shutter_mode(self, mode):
        self._shutter_mode = mode

    def get_shutter_mode(self):
        return self._shutter_mode

    def _set_image_indexing(self, indexing):
        self.image_indexing = indexing

    def _set_frame_format(self, fmt):
        self.frame_format = fmt

    def _get_attr_dev(self, attr):
        raise NotImplementedError()  # cannot be implemented.. ?

    def _set_attr_dev(self, attr, value):
        raise NotImplementedError()  # cannot be implemented.. ?

    #################################################################
    #               CAMERA PROPERTIES                               #
    #################################################################

    def get_roi(self):
        # this returns both the roi and the binning to keep them seperate for
        # easier data handling we define both seperately
        return self._roi

    def set_roi(self, roi):
        self._roi = roi
        return self._roi

    def get_hardware_binning(self):
        return self._binning

    def set_hardware_binning(self, binning):
        self._binning = binning
        return self._binning

    def get_frame_shape(self):
        roi = self.get_roi()
        self._frame_shape = [roi[1] - roi[0], roi[3] - roi[2]]
        return self._frame_shape

    def set_frame_shape(self, frame_shape):
        roi = [
            self.SensorWidth / 2 - frame_shape[0] / 2,
            self.SensorWidth / 2 + frame_shape[0] / 2,
            self.SensorHeight / 2 - frame_shape[1] / 2,
            self.SensorHeight / 2 + frame_shape[1] / 2,
        ]
        roi = [int(i) for i in roi]

        logger.info("Setting the roi to: {}", roi)
        self.set_roi(roi)
        self._frame_shape = frame_shape
        return self._frame_shape

    def update_data_size(self):
        raise NotImplementedError()

    def get_data_size(self):
        raise NotImplementedError()

    def get_exposure_time(self):
        return self._exposure_time

    def set_exposure_time(self, value):
        self._exposure_time = value

    async def start_video(self, connection):
        async def video_loop(cn: ServerConnection, stop_event: asyncio.Event):
            cn.reset_stream()
            while True:
                await asyncio.sleep(0.1)
                try:
                    if stop_event.is_set():
                        stop_event.clear()
                        logger.info("Stopping video feed")
                        break
                    ar = self.take_snapshot()
                    cn.send_stream_chunk(ar, header="video")
                    logger.debug("Sent video frame")
                except Exception as e:
                    logger.exception("Error in video loop.")
                    break

        self.video_stop_event.clear()
        self.task = asyncio.create_task(video_loop(connection, self.video_stop_event))
        await asyncio.sleep(0)  # let above start

    def stop_video(self):
        self.video_stop_event.set()
