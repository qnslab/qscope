import time

import numpy as np
from loguru import logger

from qscope.meas.decorators import requires_hardware
from qscope.system.system import SGCameraSystem
from qscope.types import (
    MAIN_CAMERA,
    PRIMARY_RF,
    SEQUENCE_GEN,
    MeasurementFrame,
    SGAndorCWESRConfig,
    SGAndorCWESRLongExpConfig,
    SGAndorPESRConfig,
)
from qscope.util.defaults import MEAS_SWEEP_TIMEOUT

from .framegrabber import FrameGrabber
from .measurement import (
    ACQ_MODE,
    NORM_MODE,
    SGCameraMeasurement,
)


@requires_hardware(
    SGCameraSystem,
    roles=(MAIN_CAMERA, SEQUENCE_GEN, PRIMARY_RF),
)
class SGAndorESRBase(SGCameraMeasurement):
    """Base class for ESR measurements using camera detection.

    Requires:
    - Camera system
    - Sequence generator
    - RF source
    """

    x_label = "Frequency (MHz)"
    y_label = "PL (counts)"
    _acq_mode = ACQ_MODE.SINGLE_MEAS
    available_ref_modes = ("", "no_rf", "fmod")
    norm_mode = NORM_MODE.DIV
    system: SGCameraSystem
    meas_config: SGAndorCWESRConfig | SGAndorPESRConfig

    def _prepare(self, continue_prev=False) -> tuple:
        # Reconnect to the RF source
        self.system.reconnect_rf()

        # Setup RF source through System method
        freq_list = self._define_rf_freq_list()

        # Calculate number of frames
        nframes = self.meas_config.avg_per_point * len(self.meas_config.sweep_x)
        if self.ref_mode != "":
            nframes *= 2

        # Setup camera through System method
        self.system.setup_camera_sequence(
            nframes=nframes,
            exposure_time=self.meas_config.exposure_time,
            frame_shape=self.meas_config.frame_shape,
        )
        camera_trig_time = self.system.get_cam_trig_time()

        logger.info(
            "Camera setup with {} frames with exposure time {} an roi of {} and a sweep length of {}",
            nframes,
            self.meas_config.exposure_time,
            self.meas_config.frame_shape,
            len(self.meas_config.sweep_x),
        )

        self.system.setup_rf_sweep(
            freq_list=freq_list,
            power=self.meas_config.rf_pow,
            step_time=camera_trig_time,
        )

        seq_settings = {
            "seq_name": self.meas_config.meas_type,
            "ref_mode": self.ref_mode,
            "exp_t": self.meas_config.exposure_time,
            "avg_per_point": self.meas_config.avg_per_point,
            "sweep_len": len(self.meas_config.sweep_x),
            "camera_trig_time": camera_trig_time,
        }
        if isinstance(self.meas_config, SGAndorPESRConfig):
            seq_settings.update(
                {
                    "laser_dur": self.meas_config.laser_dur,
                    "rf_dur": self.meas_config.rf_dur,
                }
            )

        # Setup sequence generator
        self.system.load_sequence(**seq_settings)
        logger.info("Seqgen setup")

        # start the RF source
        self.system.set_rf_state(True)
        return (
            len(self.meas_config.sweep_x),
            self.meas_config.frame_shape[0],
            self.meas_config.frame_shape[1],
        )

    def _define_rf_freq_list(self):
        if self.ref_mode == "fmod":

            # interleave the sweep_x with the fmod_freq modulated sweep
            sig, ref = (
                np.zeros(2 * len(self.meas_config.sweep_x)),
                np.zeros(2 * len(self.meas_config.sweep_x)),
            )
            idx = 0
            for f in self.meas_config.sweep_x:
                sig[idx] = f - self.meas_config.fmod_freq / 2
                ref[idx] = f + self.meas_config.fmod_freq / 2
                idx += 1
                sweep_x = np.array([f for pair in zip(sig, ref) for f in pair])
            if self.meas_config.avg_per_point > 1:
                if hasattr(self.meas_config, "long_exp") and self.meas_config.long_exp:
                    return sweep_x
                else:
                    return np.repeat(sweep_x, self.meas_config.avg_per_point)
            else:
                return sweep_x
            # return np.array([f for pair in zip(sig, ref) for f in pair])
        else:
            # if self.meas_config.avg_per_point > 1:
            #     return np.repeat(
            #         self.meas_config.sweep_x, self.meas_config.avg_per_point
            #     )
            # else:
            return self.meas_config.sweep_x

    def _reset_per_sweep(self):
        self.system.start_rf_sweep()
        self.system.reset_sequence()
        self.system.restart_camera_acquisition()
        self.system.start_sequence()

    def _stop_acq(self):
        self.system.stop_all_acquisition()

    def _single_meas_acq(self):
        self._single_acq(FrameGrabber)

    #  NOTE: example, will not be employed as attr ^ = ACQ_MODE.SINGLE_MEAS
    def _full_sweep_acq(self):
        t0 = time.time()
        while time.time() - t0 < MEAS_SWEEP_TIMEOUT:
            if self.system._seqgen.is_finished():
                break
            time.sleep(0.1)
        else:
            raise TimeoutError(
                f"Timeout ({MEAS_SWEEP_TIMEOUT}s)"
                + " waiting for measurement sequence to finish"
            )

        try:
            frames = self.system._camera.get_all_seq_frames()
            self._sweep_idx_nmeas += 1

            # assert frames.shape == (2*len(self.sweep_x) * self.meas_config.avg_per_point,
            #                         *self.meas_config.frame_shape)
            if self.meas_config.ref_mode:
                sig_frames = frames[::2]
                ref_frames = frames[1::2]
            else:
                sig_frames = frames
                ref_frames = frames[-1, 0, 0] * np.ones_like(sig_frames)

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

            if self._aoi_change_event.is_set():  # reset the rolling avg
                self._reset_rolling_avg()
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
        except Exception:
            logger.exception("Error acquiring frames, attempting to continue.")


@requires_hardware(
    SGCameraSystem,
    roles=(MAIN_CAMERA, SEQUENCE_GEN, PRIMARY_RF),
)
class SGAndorCWESR(SGAndorESRBase):
    """ESR measurement using CW excitation and camera detection."""

    _meas_config_type = SGAndorCWESRConfig
    meas_config: SGAndorCWESRConfig


@requires_hardware(
    SGCameraSystem,
    roles=(MAIN_CAMERA, SEQUENCE_GEN, PRIMARY_RF),
)
class SGAndorCWESRLongExp(SGAndorESRBase):
    """ESR measurement using CW excitation and camera detection."""

    _meas_config_type = SGAndorCWESRLongExpConfig
    meas_config: SGAndorCWESRLongExpConfig


@requires_hardware(
    SGCameraSystem,
    roles=(MAIN_CAMERA, SEQUENCE_GEN, PRIMARY_RF),
)
class SGAndorPESR(SGAndorESRBase):
    """Pulsed ESR measurement using camera detection."""

    _meas_config_type = SGAndorPESRConfig
    meas_config: SGAndorPESRConfig
