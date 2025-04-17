import asyncio
import time

import pytest
import pytest_asyncio
from loguru import logger

import qscope
import qscope.meas
import qscope.system
import qscope.util
from qscope.meas import MEAS_STATE
from qscope.types import TESTING_MEAS_CONFIG
from qscope.util import TEST_LOGLEVEL


class TestLocalMeas:
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

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_meas_logic(self, system):
        TIMEOUT = 5

        config = TESTING_MEAS_CONFIG
        qu = asyncio.Queue()
        meas = qscope.meas.MockSGAndorESR(system, config, qu)
        bg_task = asyncio.create_task(meas.state_machine())
        await asyncio.sleep(0.1)  # allow above to start

        # start measurement
        meas.start()

        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            await asyncio.sleep(0.1)
            if meas.state == MEAS_STATE.RUNNING:
                break
        else:
            raise RuntimeError("Timeout waiting for measurement to start")

        meas.pause_endsweep()

        while time.time() - t0 < 2 * TIMEOUT:
            await asyncio.sleep(0.1)
            if meas.state == MEAS_STATE.PAUSED:
                break
        else:
            raise RuntimeError("Timeout waiting for measurement to pause")

        # resume measurement
        meas.start()
        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            await asyncio.sleep(0.1)
            if meas.state == MEAS_STATE.RUNNING:
                break
        else:
            raise RuntimeError("Timeout waiting for measurement to resume")

        # stop measurement
        meas.stop_now()
        t0 = time.time()
        while t0 - time.time() < TIMEOUT:
            await asyncio.sleep(0.1)
            if meas.state == MEAS_STATE.FINISHED:
                break
        else:
            raise RuntimeError("Timeout waiting for measurement to stop")

        meas.close()

        t0 = time.time()
        while time.time() - t0 < TIMEOUT:
            await asyncio.sleep(0.1)
            if meas.state == MEAS_STATE.CLOSE:
                break
        else:
            raise RuntimeError("Timeout waiting for measurement to close")

        bg_task.cancel()
