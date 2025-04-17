from __future__ import annotations

import typing

from qscope.types import MAIN_CAMERA

if typing.TYPE_CHECKING:
    from qscope.system import SGCameraSystem

import time

import numpy as np
from loguru import logger
from numba import njit


@njit
def flatten_chunks(chunks):
    # each chunk is a 3D array, first axis is frame number
    frames = []
    for chunk in chunks:
        for frame_idx in chunk.shape[0]:
            frames.append(chunk[frame_idx, :, :])
    return frames


class FrameGrabber:
    def __init__(self, system: SGCameraSystem, *args):
        self.cache = []
        self.system = system

    # this follows the andor convention, but could have
    # subclasses for diff camera types/sdk's.
    def get_new_chunks(self):
        self.system.get_device_by_role(MAIN_CAMERA).wait_for_frame()
        frames = self.system.get_device_by_role(MAIN_CAMERA).get_all_seq_frames()
        # chunks = self.system.camera.get_all_seq_frames()
        # frames = flatten_chunks(chunks)
        self.cache.extend(frames)

    # TODO what types are we getting here??
    def get(self, *args):
        if not self.cache:
            self.get_new_chunks()
            return self.cache.pop(0)
        return self.cache.pop(0)


class MockFrameGrabber(FrameGrabber):
    def __init__(
        self, system: SGCameraSystem, rng, test_sig, test_ref, noise_sigma=0.0
    ):
        super().__init__(system)
        self.rng = rng
        self.test_sig = test_sig
        self.test_ref = test_ref
        self.noise_sigma = noise_sigma

    def get(self, sigref="sig", idx=0):
        ar = self.test_sig[idx] if sigref == "sig" else self.test_ref[idx]
        noise_ar = ar + self.rng.normal(0.0, self.noise_sigma, size=np.shape(ar))
        # convert to uint64
        ret = np.rint(noise_ar).astype(np.uint64)
        time.sleep(0.03)
        return ret
