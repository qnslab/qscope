from __future__ import annotations

import asyncio
import types
import uuid
from typing import TYPE_CHECKING, Type

import numpy as np
from loguru import logger

from qscope.meas.decorators import MeasurementRequirements, requires_hardware
from qscope.system import SGCameraSystem, System
from qscope.types import (
    MAIN_CAMERA,
    SEQUENCE_GEN,
    CameraConfig,
    MeasurementConfig,
    MeasurementFrame,
    MeasurementUpdate,
    NewMeasurement,
    PleaseWait,
    RollingAvgSweepUpdate,
    SweepUpdate,
)
from qscope.util import decimate_uneven_data

if TYPE_CHECKING:
    from qscope.meas import FrameGrabber
    from qscope.types import Notification

# ----------------
# Available States
# ----------------

MEAS_STATE = types.SimpleNamespace()
MEAS_STATE.AWAITING_START = "AWAITING_START"
MEAS_STATE.PREPARING = "PREPARING"
MEAS_STATE.RUNNING = "RUNNING"
MEAS_STATE.END_OF_SWEEP = "END_OF_SWEEP"
MEAS_STATE.PAUSED = "PAUSED"
MEAS_STATE.FINISHED = "FINISHED"
MEAS_STATE.CLOSE = "CLOSING"

ACQ_MODE = types.SimpleNamespace()
ACQ_MODE.SINGLE_MEAS = "meas"
ACQ_MODE.SINGLE_SWEEP = "sweep"

NORM_MODE = types.SimpleNamespace()
NORM_MODE.NO_NORM = "none"
NORM_MODE.DIV = "div"
NORM_MODE.SUB = "sub"
NORM_MODE.TRUE_SUB = "true_sub"


def norm_sweep(norm_mode: str, sig: np.ndarray, ref: np.ndarray):
    match norm_mode:
        case NORM_MODE.NO_NORM:
            return sig
        case NORM_MODE.DIV:
            return sig / ref
        case NORM_MODE.SUB:
            return 1 + (sig - ref) / (sig + ref)
        case NORM_MODE.TRUE_SUB:
            return (sig - ref) / np.nanmax(sig - ref)
        case _:
            raise ValueError(f"Invalid norm_mode: {norm_mode}")


class Measurement:
    # init a new Measurement instance for each individual measurement

    # subclass this for each hardware configuration/pulse sequence etc.
    # (and build up inheritance tree for hardware sysconfig)
    state = MEAS_STATE.AWAITING_START
    meas_id: str = ""
    _acq_mode: str  # ACQ_MODE -> override in subclass
    meas_config: MeasurementConfig
    _meas_config_type: Type[MeasurementConfig] = (
        MeasurementConfig  # override in subclass
    )

    _hardware_requirements: MeasurementRequirements  # overriden by decorator

    # the 'data', in general.
    sweep_x: np.ndarray  # always 1D
    sweep_y_sig: np.ndarray  # always 1D
    sweep_y_ref: np.ndarray  # always 1D
    full_y_sig: np.ndarray  # general shape
    full_y_ref: np.ndarray  # general shape
    rolling_avg_sig: np.ndarray  # always 1D
    rolling_avg_ref: np.ndarray  # always 1D

    # NB: norm is not stored or generated here.
    norm_mode: str  # NORM_MODE -> override
    ref_mode: str = ""
    available_ref_modes = ("",)  # override
    aoi: tuple[int, int, int, int] | None = None

    # these events can be set by the API methods, via async server calls
    # that are DIFFERENT to the one that starts the state machine.
    _start_event: asyncio.Event
    _stop_now_event: asyncio.Event
    _pause_endsweep_event: asyncio.Event
    _close_event: asyncio.Event
    _aoi_change_event: asyncio.Event

    _continue_prev: bool = False
    _nsweeps: int = 0
    _sweep_progress: float = 0.0  # float in [0, 100]
    _sweep_progress_increment: float = 0.0  # float in (0, 100]
    _rolling_avg_window: int = 1
    _rolling_avg_idx: int = 1
    _rolling_sum_sweep_sig: np.ndarray  # always 1D
    _rolling_sum_sweep_ref: np.ndarray  # always 1D

    _frame_sending_num: int = 2

    x_label: str = ""  # override in subclass
    y_label: str = ""  # override in subclass

    # ----------------------------------------------------------------------------------
    # =========================API (~ each has comm associated) ========================
    # ----------------------------------------------------------------------------------

    def __init__(
        self,
        system: System,
        meas_config: MeasurementConfig,
        notif_queue: asyncio.Queue[Notification],
    ):
        if not hasattr(self, "_hardware_requirements"):
            raise RuntimeError(
                "Measurement class must be decorated with `@requires_hardware`."
            )
        if meas_config.ref_mode not in self.available_ref_modes:
            raise ValueError(
                f"Invalid ref_mode {meas_config.ref_mode} for {self.__class__.__name__}. "
                + f"Available modes: {self.available_ref_modes}",
            )
        if not isinstance(meas_config, self._meas_config_type):
            raise ValueError(f"Invalid meas_config type, use {self._meas_config_type}")

        self.system = system
        self.meas_id = self._generate_meas_id()
        self.sweep_x = meas_config.sweep_x
        self.meas_config = meas_config
        self.ref_mode = meas_config.ref_mode
        self.notif_queue = (
            notif_queue  # queue to send notifications to client (server dispatches)
        )
        self.notif_queue.put_nowait(
            NewMeasurement(meas_id=self.get_meas_id(), meas_config=meas_config)
        )
        # init these
        # (can't be in class def as they need to be at *instance* level
        #  not for _all instances_ of this class)
        self._start_event = asyncio.Event()
        self._stop_now_event = asyncio.Event()
        self._pause_endsweep_event = asyncio.Event()
        self._close_event = asyncio.Event()
        self._aoi_change_event = asyncio.Event()

        # Init rolling avg stuff
        self.rolling_avg_sigs = []
        self.rolling_avg_refs = []
        self.rolling_avg_idxs = []
        self._rolling_avg_max_sweeps = 100

    async def state_machine(self):
        # start the state machine, can't be in the __init__ as it needs to be async.
        # -> keep all other fns sync.
        self.state = MEAS_STATE.AWAITING_START
        while True:
            # await asyncio.sleep(0)  # check for events
            if self.state == MEAS_STATE.CLOSE:
                logger.info("{} Measurement closing.", self.__class__.__name__)
                # may want to clear data here, object isn't going away...
                break

            # router *does stuff*, then updates what next_state should be
            try:
                next_state = await self._router(self.state)
            except Exception as e:
                logger.exception("Error in state machine.")
                next_state = MEAS_STATE.FINISHED
            if self.state != next_state:
                logger.info(
                    "Measurement state: {} {} -> {}",
                    self.__class__.__name__,
                    self.get_meas_id(),
                    next_state,
                )
                self.notif_queue.put_nowait(
                    MeasurementUpdate(
                        meas_id=self.get_meas_id(),
                        old_state=self.state,
                        new_state=next_state,
                    )
                )
            self.state = next_state
        return "ok"

    def start(self):
        if self.state not in [MEAS_STATE.PAUSED, MEAS_STATE.AWAITING_START]:
            if self._pause_endsweep_event.is_set():
                logger.debug(
                    "Waiting for meas to end sweep to pause, in START on meas_id: {}",
                    self.get_meas_id(),
                )
                raise PleaseWait("Waiting for meas to end sweep to pause.")
            else:
                logger.error(
                    "Measurement cannot be started unless paused or awaiting_start."
                )
                raise RuntimeError(
                    "Measurement cannot be started unless paused or awaiting_start."
                )
        self._continue_prev = False
        self._stop_now_event.clear()
        self._pause_endsweep_event.clear()
        self._start_event.set()

    def stop_now(self):
        # stop the measurement now, go into 'finished' state.
        self._stop_now_event.set()

    def pause_endsweep(self):
        # pause the measurement, go into 'paused' state.
        self._pause_endsweep_event.set()

    def close(self):
        if self.state not in [MEAS_STATE.PAUSED, MEAS_STATE.FINISHED]:
            if self._pause_endsweep_event.is_set() or self._stop_now_event.is_set():
                logger.debug(
                    "Waiting for meas to end sweep to pause, or stop, in CLOSE on meas_id: {}",
                    self.get_meas_id(),
                )
                raise PleaseWait("Waiting for meas to end sweep to pause, or stop.")
            else:
                logger.error("Measurement cannot be closed until paused or finished.")
                raise RuntimeError(
                    "Measurement cannot be closed until paused or finished."
                )
        self._close_event.set()

    def get_sweep(self) -> np.ndarray:
        if self.state in [MEAS_STATE.AWAITING_START]:
            nones = np.empty_like(self.sweep_x)
            nones[:] = np.nan
            return np.vstack((self.sweep_x, nones, nones))
        return np.vstack((self.sweep_x, self.sweep_y_sig, self.sweep_y_ref))

    def get_norm_mode(self):
        return self.norm_mode

    def get_frame(self, frame_type: str = "sig", frame_num: int = 2):
        raise NotImplementedError()  # only implement for camera sequences.

    def get_metadata(self):
        if self.state == MEAS_STATE.AWAITING_START:
            raise ValueError("Can only get metadata after measurement has started.")
        return self.metadata
        # TODO server needs to add the system/devices config (all the static stuff) on
        #  top of this at higher lvl

    def get_full_data(self):
        return self.full_y_sig, self.full_y_ref

    def get_meas_id(self):
        return self.meas_id

    def get_sweep_number(self):
        return self._nsweeps

    def get_sweep_progress(self):
        return self._sweep_progress

    def get_info(self):
        return {
            "meas_id": self.get_meas_id(),
            "state": self.state,
            "nsweeps": self.get_sweep_number(),
            "sweep_progress": self.get_sweep_progress(),
            "meas_config": self.meas_config.to_dict(),
        }

    def set_aoi(self, aoi: tuple[int, int, int, int] | None):
        if aoi[2] < aoi[0]:
            aoi[0], aoi[2] = aoi[2], aoi[0]
        if aoi[3] < aoi[1]:
            aoi[1], aoi[3] = aoi[3], aoi[1]
        self.aoi = aoi
        self._aoi_change_event.set()

    def get_aoi(self):
        return self.aoi

    def get_aoi_slice(self):
        if self.aoi is None:
            return slice(None), slice(None)
        return slice(self.aoi[0], self.aoi[2]), slice(self.aoi[1], self.aoi[3])

    # set frame number to send out on PUB socket
    def set_frame_num(self, frame_num: int):
        # only for camera sequences.
        raise NotImplementedError()

    def get_meas_type_name(self):
        return self.__class__.__name__

    def get_meas_save_name(self):
        return self.meas_config.save_name

    def get_norm_sweep(self):
        mode = self.get_norm_mode()
        sweep_data = self.get_sweep()
        return sweep_data[0, :], norm_sweep(mode, sweep_data[1, :], sweep_data[2, :])

    def get_rolling_avg_sweep(self):
        if self.state in [MEAS_STATE.AWAITING_START]:
            nones = np.empty_like(self.sweep_x)
            nones[:] = np.nan
            return np.vstack((self.sweep_x, nones, nones))
        return np.vstack((self.sweep_x, self.rolling_avg_sig, self.rolling_avg_ref))

    def set_rolling_avg_window(self, window: int):
        self._rolling_avg_window = window

    def get_description(self):
        return f"{self.__class__.__name__} measurement {self.get_meas_id()}"

    def get_hardware_requirements(self):
        return self._hardware_requirements

    def get_rolling_avg(self):
        return self.rolling_avg_idxs, self.rolling_avg_sigs, self.rolling_avg_refs

    # ----------------------------------------------------------------------------------
    # ================================= HELPER METHODS =================================
    # ----------------------------------------------------------------------------------

    def _start_new_sweep(self) -> bool:
        return (
            self._sweep_progress < self._sweep_progress_increment
            or self._sweep_progress < 0.001
        )

    def _generate_meas_id(self):
        self.meas_id = str(uuid.uuid4())
        # self.meas_id = "".join(
        #     random.choices(string.ascii_uppercase + string.digits, k=10)
        # )
        return self.meas_id

    def get_config_dict(self):
        return self.meas_config.to_dict()

    def _generate_metadata(self):
        metadata = {
            "meas_id": self.meas_id,
            "nsweeps": self._nsweeps,
            "measurement_name": self.__class__.__name__,
            "ref_mode": self.ref_mode,
        }
        for attr in self.meas_config.__dict__:
            metadata[attr] = getattr(self.meas_config, attr)
        self.metadata = metadata

    def _init_data(self, full_data_shape: tuple):
        # TODO set the TYPE here! (of elements) -> from meas_config, probably uint16 & uint64 or so
        self.sweep_y_sig = np.copy(self.sweep_x)
        self.sweep_y_ref = np.copy(self.sweep_x)
        self.sweep_y_sig[:] = 0
        self.sweep_y_ref[:] = 0

        self.full_y_sig = np.zeros(full_data_shape, dtype=np.uint64)
        self.full_y_ref = np.zeros(full_data_shape, dtype=np.uint64)
        self.full_y_sig[:] = 0
        self.full_y_ref[:] = 0

    async def _router(self, next_state: str) -> str:
        # NOTE: only place for sleeps!!
        match next_state:
            case MEAS_STATE.AWAITING_START:
                await asyncio.sleep(0.1)
                return self._state_awaiting_start()
            case MEAS_STATE.PREPARING:
                await asyncio.sleep(0)
                return self._state_preparing()
            case MEAS_STATE.PAUSED:
                await asyncio.sleep(0.1)
                return self._state_paused()
            case MEAS_STATE.RUNNING:
                await asyncio.sleep(0)
                return await self._state_running()
            case MEAS_STATE.END_OF_SWEEP:
                await asyncio.sleep(0)
                return self._state_end_of_sweep()
            case MEAS_STATE.FINISHED:
                await asyncio.sleep(0.1)
                return self._state_finished()
            case MEAS_STATE.CLOSE:
                await asyncio.sleep(0.1)
                return MEAS_STATE.CLOSE

    # ----------------------------------------------------------------------------------
    # ============================= STATE MACHINE - STATES =============================
    # ----------------------------------------------------------------------------------

    def _state_awaiting_start(self) -> str:
        if self._start_event.is_set():
            self._start_event.clear()
            return MEAS_STATE.PREPARING
        if self._stop_now_event.is_set():
            self._stop_now_event.clear()
            return MEAS_STATE.FINISHED
        return MEAS_STATE.AWAITING_START  # continue waiting

    def _state_preparing(self) -> str:
        self._rolling_avg_idx = 1
        nones = np.empty_like(self.sweep_x, dtype=np.float64)
        self.rolling_avg_sig = nones
        self.rolling_avg_ref = nones
        self._rolling_sum_sweep_sig = np.zeros((len(self.sweep_x),))  # define a dtype
        self._rolling_sum_sweep_ref = np.zeros((len(self.sweep_x),))

        full_data_shape = self._prepare()  # includes *2 for ref if present
        if not self._continue_prev:
            self._generate_metadata()
            self._init_data(full_data_shape)
            self._sweep_progress_increment = 100.0 / len(self.sweep_x)
            self._sweep_idx_nmeas = np.zeros(self.sweep_x.shape, dtype=int)

        self._sweep_progress = 0.0
        self._reset_per_sweep()  # run before first acq.
        return MEAS_STATE.RUNNING

    async def _state_running(self) -> str:
        try:
            # this measurement type will acquire the full sweep in one method call.
            if self._acq_mode == ACQ_MODE.SINGLE_SWEEP:
                self._full_sweep_acq()
                return MEAS_STATE.END_OF_SWEEP

            # this measurement type will acquire one meas (sig/ref pair * avg_per_point)
            elif self._acq_mode == ACQ_MODE.SINGLE_MEAS:
                self._single_meas_acq()
                self._sweep_progress += self._sweep_progress_increment
                if self._sweep_progress + self._sweep_progress_increment > 100.0:
                    return MEAS_STATE.END_OF_SWEEP

                self.notif_queue.put_nowait(
                    SweepUpdate(
                        meas_id=self.get_meas_id(),
                        sweep_progress=self._sweep_progress,
                        nsweeps=self._nsweeps,
                        aoi=self.aoi,
                        sweep_data=self.get_sweep(),
                    )
                )
        except Exception:
            logger.exception("Error while RUNNING meas {}.", self.get_meas_id())
            self._stop_acq()
            return MEAS_STATE.FINISHED
        if self._stop_now_event.is_set():
            self._stop_acq()
            self._stop_now_event.clear()
            return MEAS_STATE.FINISHED
        return MEAS_STATE.RUNNING

    def _state_paused(self) -> str:
        if self._start_event.is_set():
            self._continue_prev = True
            self._start_event.clear()
            return MEAS_STATE.PREPARING
        if self._stop_now_event.is_set():
            self._stop_now_event.clear()
            return MEAS_STATE.FINISHED
        return MEAS_STATE.PAUSED

    def _state_end_of_sweep(self) -> str:
        """Handle end of sweep state."""
        # Increment sweep counter
        self._nsweeps += 1

        # Send sweep update notification
        self.notif_queue.put_nowait(
            SweepUpdate(
                meas_id=self.get_meas_id(),
                sweep_progress=100.0,
                nsweeps=self._nsweeps,
                aoi=self.aoi,
                sweep_data=self.get_sweep(),
            )
        )

        # Handle rolling average updates
        if self._rolling_avg_idx == self._rolling_avg_window:
            self._rolling_avg_idx = 1
            self.rolling_avg_sigs.append(
                (self._rolling_sum_sweep_sig / self._rolling_avg_window)
            )
            self.rolling_avg_refs.append(
                (self._rolling_sum_sweep_ref / self._rolling_avg_window)
            )
            self.rolling_avg_idxs.append(self._nsweeps)
            self._rolling_sum_sweep_sig[:] = 0
            self._rolling_sum_sweep_ref[:] = 0
            if len(self.rolling_avg_idxs) > self._rolling_avg_max_sweeps:
                # interpolate/decimate
                new_idxs, self.rolling_avg_sigs = decimate_uneven_data(
                    self.rolling_avg_idxs, self.rolling_avg_sigs, 2
                )
                _, self.rolling_avg_refs = decimate_uneven_data(
                    self.rolling_avg_idxs, self.rolling_avg_refs, 2
                )
                self.rolling_avg_idxs = new_idxs
            self.notif_queue.put_nowait(
                RollingAvgSweepUpdate(
                    meas_id=self.get_meas_id(),
                    avg_window=self._rolling_avg_window,
                    aoi=self.aoi,
                    sweep_idxs=np.asarray(self.rolling_avg_idxs),
                    sweep_x=self.sweep_x,
                    sweep_ysig=np.asarray(self.rolling_avg_sigs),
                    sweep_yref=np.asarray(self.rolling_avg_refs),
                )
            )
        else:
            self._rolling_avg_idx += 1

        # Reset for next sweep
        self._reset_per_sweep()

        # Check if we should pause
        if self._pause_endsweep_event.is_set():
            self._stop_acq()
            self._pause_endsweep_event.clear()
            return MEAS_STATE.PAUSED

        # Otherwise continue running
        self._sweep_progress = 0.0
        return MEAS_STATE.RUNNING

    def _state_finished(self) -> str:
        # can't leave finished state.
        # have sleeps to allow for access to data etc.
        if self._close_event.is_set():
            self._close_event.clear()
            return MEAS_STATE.CLOSE
        return MEAS_STATE.FINISHED

    # ----------------------------------------------------------------------------------
    # ============================== METHOD IMPLEMENTATION =============================
    # ----------------------------------------------------------------------------------

    def _prepare(self):
        raise NotImplementedError()

    def _stop_acq(self):
        # stop any hardware (e.g. seqgen, camera) that is currently running.
        raise NotImplementedError()

    def _single_meas_acq(self):
        # run a single measurement, e.g. acquire a single sig frame
        # also in charge of: initing hardware for the sweep
        # needs to update the data attrs: sweep_x, sweep_y_sig, sweep_y_ref
        # and full_y_sig, full_y_ref.
        # all required configuration is contained in the meas_config attribute.
        # TODO check example and see what this *needs to set*
        # (can any of it be wrapped??)
        # NO CHANGE means single-point, but can be multiple frames e.g. avg-per-point
        # ALSO needs to do the rolling avg stuff.
        raise NotImplementedError()

    def _full_sweep_acq(self):
        # setup hardware for sweep, then run full sweep, updating data attrs:
        # sweep_x, sweep_y_sig, sweep_y_ref and full_y_sig, full_y_ref.
        # all required configuration is contained in the meas_config attribute.
        raise NotImplementedError()

    def _reset_per_sweep(self):
        # reset hardware per-sweep etc.
        raise NotImplementedError()


@requires_hardware(SGCameraSystem, roles=(MAIN_CAMERA, SEQUENCE_GEN))
class SGCameraMeasurement(Measurement):
    meas_config: CameraConfig
    system: SGCameraSystem

    def get_frame(self, frame_num: int = 2, frame_type: str = "sig"):
        if self.state in [MEAS_STATE.AWAITING_START]:
            nones = np.empty(self.meas_config.frame_shape)
            nones[:] = np.nan
            return nones
        if frame_type not in ["sig", "ref"]:
            raise ValueError("frame_type must be 'sig' or 'ref'")
        data = self.full_y_sig if frame_type == "sig" else self.full_y_ref
        return data[frame_num, :, :]

    def set_frame_num(self, frame_num: int):
        self._frame_sending_num = frame_num
        if hasattr(self, "full_y_sig"):
            self.notif_queue.put_nowait(
                MeasurementFrame(
                    meas_id=self.meas_id,
                    frame_num=frame_num,
                    sig_frame=self.full_y_sig[frame_num, :, :],
                    ref_frame=self.full_y_ref[frame_num, :, :],
                )
            )

    # meas_config must have: frame_shape, avg_per_point.
    def _single_acq(self, framegrabber_type: Type[FrameGrabber], *args):
        if self._start_new_sweep():
            self._idx = 0
            self.framegrabber = framegrabber_type(self.system, *args)
        else:
            self._idx += 1
        self._sweep_idx_nmeas[self._idx] += 1

        sig_acc_frame = np.zeros(self.meas_config.frame_shape, dtype=np.uint64)
        ref_acc_frame = np.zeros(self.meas_config.frame_shape, dtype=np.uint64)

        # For in the case of long exposure acquisition, we need to average the 
        # frames all from sig and all from ref rather than interleaving.

        if hasattr(self.meas_config, "long_exp") and self.meas_config.long_exp:
            for _ in range(self.meas_config.avg_per_point):
                frame = self.framegrabber.get("sig", self._idx)
                sig_acc_frame += frame
            for _ in range(self.meas_config.avg_per_point):
                frame = self.framegrabber.get("ref", self._idx)
                ref_acc_frame += frame
        else:
            # Assume a normal interleaved acquisition
            # e.g. sig, ref, sig, ref, ...
            for _ in range(self.meas_config.avg_per_point):
                for sigref in ("sig", "ref") if self.meas_config.ref_mode else ("sig",):
                    frame = self.framegrabber.get(sigref, self._idx)
                    avg_frame = sig_acc_frame if sigref == "sig" else ref_acc_frame
                    avg_frame += frame
            if not self.meas_config.ref_mode:
                ref_acc_frame = sig_acc_frame.copy()

            if self._idx == self._frame_sending_num:
                self.notif_queue.put_nowait(
                    MeasurementFrame(
                        meas_id=self.meas_id,
                        frame_num=self._idx,
                        sig_frame=sig_acc_frame,
                        ref_frame=ref_acc_frame,
                    )
                )

        # now update the data arrays
        self.full_y_sig[self._idx, :, :] += sig_acc_frame
        self.full_y_ref[self._idx, :, :] += ref_acc_frame

        aoi_slc = self.get_aoi_slice()

        self._rolling_sum_sweep_sig[self._idx] += np.sum(
            sig_acc_frame[aoi_slc[0], aoi_slc[1]], axis=(0, 1)
        )
        self._rolling_sum_sweep_ref[self._idx] += np.sum(
            ref_acc_frame[aoi_slc[0], aoi_slc[1]], axis=(0, 1)
        )

        # need to update sweep avg for full sweep, as aoi has changed
        if self._aoi_change_event.is_set():
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
        else:  # quicker in other cases: just update this _idx
            with np.errstate(
                divide="ignore", invalid="ignore"
            ):  # happy to set 1/0 to NaN
                self.sweep_y_sig[self._idx] = (
                    np.sum(
                        self.full_y_sig[self._idx, aoi_slc[0], aoi_slc[1]], axis=(0, 1)
                    )
                    / self._sweep_idx_nmeas[self._idx]
                )
                self.sweep_y_ref[self._idx] = (
                    np.sum(
                        self.full_y_ref[self._idx, aoi_slc[0], aoi_slc[1]], axis=(0, 1)
                    )
                    / self._sweep_idx_nmeas[self._idx]
                )

    def get_integrated_image(self):
        return self.full_y_sig.mean(axis=0)
