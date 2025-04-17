"""
Test system logic, without the server comms etc. complicating things.
"""

import numpy as np
import pytest
import pytest_asyncio
from loguru import logger

import qscope
import qscope.system
import qscope.util
from qscope.util import TEST_LOGLEVEL


class TestLocalSystem:
    @pytest.fixture(autouse=True, scope="class")
    def client_log(self):
        qscope.util.start_client_log(
            log_level=TEST_LOGLEVEL, log_to_stdout=True, log_to_file=True
        )
        yield
        qscope.util.shutdown_client_log()

    @pytest.fixture(autouse=True, scope="function")
    def log(self, request):
        logger.warning("STARTED Test '{}'".format(request.node.originalname))

        def fin():
            logger.warning("COMPLETED Test '{}' \n".format(request.node.originalname))

        request.addfinalizer(fin)

    @pytest.fixture(scope="class")
    def system(self):
        sys = qscope.system.SGCameraSystem("mock")
        sys.startup()
        yield sys
        sys.packdown()

    def test_mockcam_setup(self, system: qscope.system.SGCameraSystem):
        assert system.has_camera() is True

    def test_mockcam_frame(self, system: qscope.system.SGCameraSystem):
        frame_shape = system.get_frame_shape()
        assert np.all(
            [i == j for i, j in zip(np.shape(system.take_snapshot()), frame_shape)]
        )

    def test_camera_snapshot(self, system: qscope.system.SGCameraSystem):
        system.set_camera_params(exp_t=0.1, image_size=(2560, 2560), binning=(1, 1))

        frame_shape = system.get_frame_shape()
        frame = system.take_snapshot()
        assert np.all([i == j for i, j in zip(np.shape(frame), frame_shape)])
