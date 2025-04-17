import time
from dataclasses import dataclass

import numpy as np
from loguru import logger
from numba import njit

from qscope.system import SGCameraSystem
from qscope.types import MAIN_CAMERA, MeasurementFrame, MockSGAndorESRConfig

from ..device import MockCamera
from .decorators import requires_hardware
from .framegrabber import MockFrameGrabber
from .measurement import (
    ACQ_MODE,
    NORM_MODE,
    SGCameraMeasurement,
)


# pulsed here means we use a seqgen, not that is isn't CW.
@requires_hardware(SGCameraSystem, roles=(MAIN_CAMERA,))
class MockSGAndorESR(SGCameraMeasurement):
    x_label = "Frequency (MHz)"
    y_label = "PL (counts)"
    _acq_mode = ACQ_MODE.SINGLE_MEAS
    # _acq_mode = ACQ_MODE.SINGLE_SWEEP # tested: works, yay
    _meas_config_type = MockSGAndorESRConfig
    meas_config: MockSGAndorESRConfig  # I've overriden just to give my IDE type info
    available_ref_modes = (
        "",
        "no_rf",
    )  # TODO match 'real' esr config once that's written
    norm_mode = NORM_MODE.DIV
    rng = np.random.default_rng(12345)

    def _prepare(self, continue_prev=False) -> tuple[int, ...]:
        # FIXME need to restrict the type of this (here & in real impl)
        # are we setting the camera bits?
        # also need to set frame_format ('try_chunks')
        self._test_sig, self._test_ref = self._get_data_cube()
        return len(self.sweep_x), *self.meas_config.frame_shape

    def _stop_acq(self):
        # don't need to do anything here
        pass

    def _reset_per_sweep(self):
        pass

    def _single_meas_acq(self):
        self._single_acq(
            MockFrameGrabber,
            self.rng,
            self._test_sig,
            self._test_ref,
            self.meas_config.noise_sigma,
        )

    def _get_data_cube(self):
        """
        generate a 2-peak lorentzian data cube:
        two cocentric semicircles of opposite magnetic field"""

        @njit
        def lorentz(x, params):
            """lorentzian defined by fwhm, peak position, height"""
            w, x0, A = params
            R = ((x - x0) / (w / 2)) ** 2  # w/2 to convert from fwhm to hwhm
            return A * (1 / (1 + R))

        @njit("void(uint64[:, :, :], float64[:], float64[:])")
        def apply_lorentz(raw_cube, freqs, params):
            for ypos in range(raw_cube.shape[1]):
                for xpos in range(raw_cube.shape[2]):
                    raw_cube[:, ypos, xpos] = 50.0 * (lorentz(freqs, params) + 1.0)

        raw_cube = np.empty(
            (len(self.sweep_x), *self.meas_config.frame_shape), dtype=np.uint64
        )
        freqs = self.sweep_x
        apply_lorentz(raw_cube, freqs, np.array([50, 2870, -0.4], dtype=np.float64))
        return raw_cube, raw_cube[0, 0, 0] * np.ones_like(
            raw_cube
        )  # ref is just a flat line

    def _full_sweep_acq(self):
        try:
            frames = np.empty(
                (2 * len(self.sweep_x), *self.meas_config.frame_shape), dtype=np.uint16
            )
            frames[0::2] = self._test_sig
            frames[1::2] = self._test_ref
            self._sweep_idx_nmeas += 1

            assert frames.shape == (
                2 * len(self.sweep_x) * self.meas_config.avg_per_point,
                *self.meas_config.frame_shape,
            )
            if self.meas_config.ref_mode:
                sig_frames = frames[::2]
                ref_frames = frames[1::2]
            else:
                sig_frames = frames
                ref_frames = frames[-1, 0, 0] * np.ones_like(
                    sig_frames, dtype=np.uint16
                )

            # reduce avg_per_point
            sig_frames = np.sum(
                sig_frames.reshape(
                    -1, self.meas_config.avg_per_point, *self.meas_config.frame_shape
                ),
                axis=1,
            )
            ref_frames = np.sum(
                ref_frames.reshape(
                    -1, self.meas_config.avg_per_point, *self.meas_config.frame_shape
                ),
                axis=1,
            )

            # update data attrs
            self.full_y_sig += sig_frames
            self.full_y_ref += ref_frames

            aoi_slc = self.get_aoi_slice()

            self._rolling_sum_sweep_sig += np.sum(
                sig_frames[:, aoi_slc[0], aoi_slc[1]], axis=(1, 2)
            )
            self._rolling_sum_sweep_ref += np.sum(
                ref_frames[:, aoi_slc[0], aoi_slc[1]], axis=(1, 2)
            )

            if (
                self._aoi_change_event.is_set()
            ):  # not used for this mode, but do clear the event
                self._aoi_change_event.clear()
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):  # happy to set 1/0 to NaN
                self.sweep_y_sig = (
                    np.sum(self.full_y_sig[:, aoi_slc[0], aoi_slc[1]], axis=(1, 2))
                    / self._sweep_idx_nmeas
                )
                self.sweep_y_ref = (
                    np.sum(self.full_y_ref[:, aoi_slc[0], aoi_slc[1]], axis=(1, 2))
                    / self._sweep_idx_nmeas
                )

            # send Notifications
            self.notif_queue.put_nowait(
                MeasurementFrame(
                    meas_id=self.meas_id,
                    frame_num=self._frame_sending_num,
                    sig_frame=sig_frames[self._frame_sending_num, :, :],
                    ref_frame=ref_frames[self._frame_sending_num, :, :],
                )
            )
        except Exception as e:
            logger.exception("Error acquiring frames, attempting to continue.")
