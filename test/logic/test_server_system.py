import asyncio
import os
import pathlib

import numpy as np
import pytest
import pytest_asyncio
from loguru import logger

import qscope
import qscope.server
import qscope.util
from qscope.server import close_connection
from qscope.system import SGCameraSystem
from qscope.types import TESTING_MEAS_CONFIG, ClientSyncResponse, CommsError
from qscope.util import TEST_LOGLEVEL


class TestServerSystemUpDown:
    client_sync: ClientSyncResponse

    @pytest_asyncio.fixture(autouse=True, scope="class", loop_scope="class")
    def client_log(self):
        qscope.util.start_client_log(
            log_level=TEST_LOGLEVEL, log_to_stdout=True, log_to_file=True
        )
        yield
        qscope.util.shutdown_client_log()

    @pytest_asyncio.fixture(autouse=True, scope="function", loop_scope="class")
    def log(self, request):
        logger.warning("STARTED Test '{}'".format(request.node.originalname))

        def fin():
            logger.warning("COMPLETED Test '{}' \n".format(request.node.originalname))

        request.addfinalizer(fin)

    # noinspection PyArgumentList
    @pytest_asyncio.fixture(loop_scope="class", scope="class")
    async def client_connection(self):
        proc, client_connection, client_sync = qscope.server.start_bg(
            "mock",
            log_level=TEST_LOGLEVEL,
        )
        TestServerSystemUpDown.client_sync = client_sync  # store for test later
        yield client_connection
        qscope.server.close_bg(proc, client_connection)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_mock_updown(self, client_connection):
        logger.info("==> testing updown")

    @pytest.mark.asyncio
    async def test_client_sync(self, client_connection):
        assert "MockCamera_1" in TestServerSystemUpDown.client_sync.sys_metadata.keys()
        assert (
            "MainCamera"
            in TestServerSystemUpDown.client_sync.sys_metadata["MockCamera_1"]["roles"]
        )

    def test_client_log_path(self):
        clp = qscope.util.get_log_filename()
        assert clp == qscope.util.log_default_path_client()

    def test_custom_client_log_path(self):
        clp = qscope.util.get_log_filename()
        assert os.path.normpath(clp) == str(
            pathlib.Path.home().joinpath(".qscope/client.log")
        )

    @pytest.mark.asyncio
    async def test_server_log_path(self, client_connection):
        slp = qscope.server.get_server_log_path(client_connection)
        assert slp == qscope.util.log_default_path_server()

    @pytest.mark.asyncio
    async def test_multi_message(self, client_connection):
        messages = ["hello", "world", "foo", "bar"]
        for msg in messages:
            assert qscope.server.echo(client_connection, msg) == msg


class TestMockCameraServer:
    # This one actually starts-up the system.
    @pytest_asyncio.fixture(autouse=True, scope="class", loop_scope="class")
    def client_log(self):
        qscope.util.start_client_log(
            log_level=TEST_LOGLEVEL, log_to_stdout=True, log_to_file=True
        )
        yield
        qscope.util.shutdown_client_log()

    @pytest_asyncio.fixture(autouse=True, scope="function", loop_scope="class")
    def log(self, request):
        logger.trace("STARTED Test '{}'".format(request.node.originalname))

        def fin():
            logger.trace("COMPLETED Test '{}' \n".format(request.node.originalname))

        request.addfinalizer(fin)

    # noinspection PyArgumentList
    @pytest_asyncio.fixture(loop_scope="class", scope="class")
    async def client_connection(self):
        proc, client_connection, client_sync = qscope.server.start_bg(
            "mock",
            log_level=TEST_LOGLEVEL,
        )
        qscope.server.startup(client_connection)
        yield client_connection
        qscope.server.packdown(client_connection)
        qscope.server.close_bg(proc, client_connection)

    @pytest_asyncio.fixture(scope="class", loop_scope="class")
    async def additional_client_connections(self):
        client_connections = []
        for _ in range(4):
            client, _ = qscope.server.open_connection()
            client_connections.append(client)
        yield client_connections
        for client in client_connections:
            close_connection(client)

    @pytest.mark.asyncio
    async def test_snapshot_comms(self, client_connection):
        qscope.server.camera_set_params(
            client_connection, exp_t=0.1, image_size=(2560, 2560), binning=(1, 1)
        )
        frame_shape = qscope.server.camera_get_frame_shape(client_connection)
        frame = qscope.server.camera_take_snapshot(client_connection)
        assert np.all([i == j for i, j in zip(np.shape(frame), frame_shape)])

    @pytest.mark.asyncio
    async def test_fail_lock(self, client_connection):
        tsk, qu = qscope.server.start_bg_notif_listener(client_connection)
        meas_id = qscope.server.add_measurement(client_connection, TESTING_MEAS_CONFIG)
        await asyncio.sleep(0.1)
        qscope.server.camera_start_video(client_connection)
        await asyncio.sleep(0.1)
        with pytest.raises(CommsError):
            qscope.server.start_measurement_wait(client_connection, meas_id)
        tsk.cancel()
        qscope.server.clean_queue(qu)

        qscope.server.camera_stop_video(client_connection)
        qscope.server.stop_measurement(client_connection, meas_id)
        qscope.server.close_measurement_wait(client_connection, meas_id)

    @pytest.mark.asyncio
    async def test_multiple_connections(
        self, client_connection, additional_client_connections
    ):
        msgs = ["hello", "world", "foo", "bar"]
        for client, msg in zip(additional_client_connections, msgs):
            echo = qscope.server.echo(client, msg)
            assert echo == msg

    @pytest.mark.asyncio(loop_scope="class")
    async def test_multiclient_notifs(
        self, client_connection, additional_client_connections
    ):
        tsks = []
        qus = []
        meas_ids = []
        for client in additional_client_connections:
            tsk, qu = qscope.server.start_bg_notif_listener(client)
            tsks.append(tsk)
            qus.append(qu)
        await asyncio.sleep(0.1)
        for client in additional_client_connections:
            meas_id = qscope.server.add_measurement(client, TESTING_MEAS_CONFIG)
            meas_ids.append(meas_id)
        assert len(meas_ids) > 0
        await asyncio.sleep(0.1)
        for qu in qus:
            assert qu.qsize() > 0
        for tsk in tsks:
            tsk.cancel()

        for i, client in enumerate(additional_client_connections):
            qscope.server.stop_measurement(client, meas_ids[i])
            qscope.server.close_measurement_wait(client, meas_ids[i])
