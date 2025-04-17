"""Message types and constants for client-server communication."""

from __future__ import annotations

import pickle
import types
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from mashumaro.mixins.msgpack import DataClassMessagePackMixin
from mashumaro.types import Discriminator


def get_all_subclasses_map(cls: type) -> dict[str:type]:
    """Get all subclasses of a class recursively."""

    def _get_all(clas: type, subclasses: dict[str:type]):
        if not clas.__subclasses__():
            return subclasses
        for subcls in clas.__subclasses__():
            subclasses[subcls.__name__] = subcls
            subclasses |= _get_all(subcls, subclasses)
        return subclasses

    return _get_all(cls, dict())


@dataclass
class Message(DataClassMessagePackMixin):
    """Base class for all messages."""

    def __repr__(self):
        msg = self.__class__.__name__ + "("
        for i, (field, val) in enumerate(self.__dict__.items()):
            if i not in (0, len(self.__dict__)):
                msg += ", "
            if isinstance(val, np.ndarray):
                msg += f"{field}=<Array>"
            else:
                msg += f"{field}={getattr(self, field)}"
        return msg + ")"


@dataclass(repr=False)
class Request(Message):
    """A request from client to server.

    Request needs to be general (client->server) as server has no info on what type of
    message it's getting. Requester does know what type of response to expect, so
    that object can be specialised.
    """

    command: str
    params: dict[str, bool | str | float | int | tuple | None] = field(
        default_factory=dict
    )


@dataclass(kw_only=True, repr=False)
class Response(Message):
    """A response from server to client's request.
    ALL Responses have an error string, or empty if no error"""

    type: str  # subclass to define
    value: Any  # subclass to define

    class Config:
        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass(kw_only=True, repr=False)
class MsgResponse(Response):
    type: str = "msg"
    value: str = ""


@dataclass(kw_only=True, repr=False)
class ArrayResponse(Response):
    type: str = "array"
    value: np.ndarray = field(
        metadata={"serialize": pickle.dumps, "deserialize": pickle.loads},
        default_factory=lambda: np.array([]),
    )


# TODO change all dicts to DataClasses.
@dataclass(kw_only=True, repr=False)
class DictResponse(Response):
    type: str = "dict"
    value: dict = field(default_factory=dict)


@dataclass(kw_only=True, repr=False)
class ValueResponse(Response):
    type: str = "value"
    value: int | float | str | bool = False


@dataclass(kw_only=True, repr=False)
class Shape2DResponse(Response):
    type: str = "shape2D"
    value: tuple[int, int] = (-1, -1)


@dataclass(kw_only=True, repr=False)
class TupleResponse(Response):
    type: str = "tuple"
    value: tuple = ()


@dataclass(kw_only=True, repr=False)
class ErrorResponse(Response):
    type: str = "error"
    value: str = ""


@dataclass(kw_only=True, repr=False)
class WarningResponse(Response):
    type: str = "warning"
    value: str = ""


@dataclass(kw_only=True, repr=False)
class ClientSyncResponse(Response):
    type: str = "client_sync"
    system_type: str
    system_name: str
    is_streaming: bool
    all_meas_info: dict
    sys_metadata: dict
    version: str
    hardware_started_up: bool


@dataclass(kw_only=True, repr=False)
class PleaseWaitResponse(Response):
    type: str = "please_wait"
    value: str = ""


@dataclass(kw_only=True, repr=False)
class Notification(Message):
    type: str

    class Config:
        discriminator = Discriminator(field="type", include_subtypes=True)


@dataclass(kw_only=True, repr=False)
class SweepUpdate(Notification):
    type: str = "sweep_update"
    meas_id: str
    sweep_progress: float
    nsweeps: int
    aoi: tuple[int, int, int, int] | None
    sweep_data: np.ndarray = field(
        metadata={"serialize": pickle.dumps, "deserialize": pickle.loads}
    )


@dataclass(kw_only=True, repr=False)
class RollingAvgSweepUpdate(Notification):
    type: str = "rolling_avg_sweep_update"
    meas_id: str
    avg_window: int
    aoi: tuple[int, int, int, int] | None
    # plot will be a 2d color mesh (y, x) = (n_sweeps, n_points) [origin=upper right]
    sweep_idxs: np.ndarray = field(
        metadata={"serialize": pickle.dumps, "deserialize": pickle.loads}
    )  # 1D array shape: n_sweeps
    sweep_x: np.ndarray = field(
        metadata={"serialize": pickle.dumps, "deserialize": pickle.loads}
    )  # 1D array, shape: n_points
    sweep_ysig: np.ndarray = field(
        metadata={"serialize": pickle.dumps, "deserialize": pickle.loads}
    )  # 2D array (n_sweeps, n_points)
    sweep_yref: np.ndarray = field(
        metadata={"serialize": pickle.dumps, "deserialize": pickle.loads}
    )  # 2D array (n_sweeps, n_points)


@dataclass(kw_only=True, repr=False)
class MeasurementUpdate(Notification):
    type: str = "measurement_update"
    meas_id: str
    old_state: str
    new_state: str


@dataclass(kw_only=True, repr=False)
class NewMeasurement(Notification):
    type: str = "new_measurement"
    meas_id: str
    meas_config: Any = field(
        metadata={"serialize": pickle.dumps, "deserialize": pickle.loads}
    )


@dataclass(kw_only=True, repr=False)
class NewStream(Notification):
    type: str = "new_stream"
    stream_type: str


@dataclass(kw_only=True, repr=False)
class SaveFullComplete(Notification):
    """Notification sent when a full data save operation completes"""

    type: str = "save_full_complete"
    meas_id: str
    save_path: str


@dataclass(kw_only=True, repr=False)
class MeasurementFrame(Notification):
    type: str = "measurement_frame"
    meas_id: str
    frame_num: int
    sig_frame: np.ndarray = field(
        metadata={"serialize": pickle.dumps, "deserialize": pickle.loads}
    )
    ref_frame: np.ndarray = field(
        metadata={"serialize": pickle.dumps, "deserialize": pickle.loads}
    )
