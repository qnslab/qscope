# test that we can setup/open a real hardware system.

import pytest
import pytest_asyncio
from loguru import logger

import qscope
import qscope.meas
import qscope.server
import qscope.util
from qscope.system import SGCameraSystem
from qscope.util import TEST_LOGLEVEL

# test hardware stuff WITHOUT server comms. (more useful for debugging)


class TestLocalAndorSetup:
    @pytest.fixture()
    def client_log(self):
        qscope.util.start_client_log(log_to_file=True, log_level=TEST_LOGLEVEL)
        yield
        qscope.util.shutdown_client_log()

    # noinspection PyArgumentList
    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.usefixtures("client_log")
    async def test_local_open(self):
        system = SGCameraSystem("hqdm")
        dev_status = system.startup()
        assert all([dev["status"] for dev in dev_status.values()])
        system.packdown()
