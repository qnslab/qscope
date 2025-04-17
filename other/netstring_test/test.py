import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from mashumaro.mixins.msgpack import DataClassMessagePackMixin
import pickle

@dataclass
class Message(DataClassMessagePackMixin):
    """"""
    
@dataclass
class ArrayResponse(Message):
    arr: npt.NDArray = field(
        metadata={"serialize": pickle.dumps, "deserialize": pickle.loads}
    )


x = np.arange(9).reshape((3,3))
print(x)

y = ArrayResponse(x)

print(y)

z = ArrayResponse.from_msgpack(y.to_msgpack())

print(z)

@dataclass
class Request(Message):
    """A request from client of server.
    Request needs to be general (client->server) as server has no info on what type of
    message it's getting.
    Requester does know what type of response to expect, so
    that object can be specialised.
    """
    command: str
    params: dict

rqst = Request("hello", {"blaa":"blaa", "image_size": (2560, 2560)})

byte_request = rqst.to_msgpack()
print(byte_request)

rqst2 = Request("hello", {"blaa":"blaa", "image_size": "\n"})
br2 = rqst2.to_msgpack()
print(br2)

request = Request.from_msgpack(byte_request)

print(request)

END = b","
END_ORD = ord(END)


def encode(payload):
    """
    Convert a payload (bytes) to a netstring frame.
    """
    return f"{len(payload)}:".encode() + payload + END


def decode(frame):
    """
    Retrieve payload from the frame (bytes). Frame must be a complete netstring
    frame otherwise ValueError exception is raised.
    """
    ndig = frame.index(b":")
    n = int(frame[0:ndig])
    start = ndig + 1
    end = start + n + 1
    if len(frame) < end:
        raise ValueError("Incomplete frame")
    result = frame[start : end - 1]
    if frame[end - 1] != END_ORD:
        raise ValueError("Received frame with invalid format")
    return result

# NOICE eh
bre = encode(byte_request)
print(decode(bre))
print(byte_request)
print(bre)
