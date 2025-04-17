# test that we can setup/open a real hardware system.

import pytest
import pytest_asyncio
from loguru import logger

import qscope
import qscope.meas
import qscope.server
import qscope.util
from qscope.util import TEST_LOGLEVEL


class TestAndorSetup:
    # noinspection PyArgumentList
    @pytest_asyncio.fixture(loop_scope="class", scope="class")
    async def connection(self):
        proc, connection, _ = qscope.server.start_bg(
            "hqdm",
            log_level=TEST_LOGLEVEL,
        )
        yield connection
        qscope.server.close_bg(proc, connection)

    @pytest.fixture()
    def client_log(self):
        qscope.util.start_client_log(log_to_file=True, log_level=TEST_LOGLEVEL)
        yield
        qscope.util.shutdown_client_log()

    # noinspection PyArgumentList
    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.usefixtures("client_log")
    async def test_server_open(self, connection):
        assert qscope.server.echo(connection, "ping") == "ping"
        ok, dev_status = qscope.server.startup(connection)
        assert ok
        qscope.server.packdown(connection)
