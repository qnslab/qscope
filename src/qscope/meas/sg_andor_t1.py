from qscope.system.system import SGCameraSystem
from qscope.types import MAIN_CAMERA, PRIMARY_RF, SEQUENCE_GEN, SGAndorT1Config

from .decorators import requires_hardware
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
class SGAndorT1(SGCameraMeasurement):
    x_label = "Time (s)"
    y_label = "PL (counts)"
    _acq_mode = ACQ_MODE.SINGLE_MEAS
    _meas_config_type = SGAndorT1Config
    meas_config: SGAndorT1Config  # I've overriden just to give my IDE type info
    available_ref_modes = ("", "π at end", "π at start")
    norm_mode = NORM_MODE.DIV

    def _prepare(self, continue_prev=False) -> tuple:
        # SETUP RF SOURCE
        self.system.setup_single_rf_freq(
            self.meas_config.rf_freq, self.meas_config.rf_pow
        )

        nframes = self.meas_config.avg_per_point * len(self.meas_config.sweep_x)
        # camera needs to know about ref here
        if self.ref_mode == "no_rf":
            nframes = nframes * 2

        # Setup camera through System method
        self.system.setup_camera_sequence(
            nframes=nframes,
            exposure_time=self.meas_config.exposure_time,
            frame_shape=self.meas_config.frame_shape,
        )
        camera_trig_time = self.system.get_cam_trig_time()

        # LOAD THE PULSEBLASTER SEQUENCE
        self.system.load_sequence(
            seq_name=self.meas_config.meas_type,
            ref_mode=self.ref_mode,
            exp_t=self.meas_config.exposure_time,
            avg_per_point=self.meas_config.avg_per_point,
            laser_dur=self.meas_config.laser_dur,
            pi_dur=self.meas_config.pi_dur,
            sweep_x=self.meas_config.sweep_x,
            laser_delay=self.meas_config.laser_delay,
            rf_delay=self.meas_config.rf_delay,
            rf_pow=self.meas_config.rf_pow,
            rf_freq=self.meas_config.rf_freq,
            camera_trig_time=camera_trig_time,
        )
        self.system.set_rf_state(True)
        return (
            len(self.meas_config.sweep_x),
            self.meas_config.frame_shape[0],
            self.meas_config.frame_shape[1],
        )

    def _reset_per_sweep(self):
        self.system.reset_sequence()
        self.system.restart_camera_acquisition()
        self.system.start_sequence()

    def _stop_acq(self):
        # stop any hardware (e.g. seqgen, camera) that is currently running.
        self.system.stop_all_acquisition()

    def _single_meas_acq(self):
        self._single_acq(FrameGrabber)  # see _single_acq in `PulsedCameraMeasurement`
