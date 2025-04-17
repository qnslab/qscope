import asyncio

import numpy as np
import pytest
import pytest_asyncio
from loguru import logger

import qscope
import qscope.server
import qscope.util
from qscope.meas import MEAS_STATE, Measurement, MockSGAndorESRConfig
from qscope.server import close_connection
from qscope.server.client import wait_for_notif
from qscope.types import (
    TESTING_MEAS_CONFIG,
    MeasurementFrame,
    MeasurementUpdate,
    NewMeasurement,
    NewStream,
    RollingAvgSweepUpdate,
    SweepUpdate,
)
from qscope.util import TEST_LOGLEVEL


class TestServerNotifs:
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

    @pytest_asyncio.fixture(loop_scope="class", scope="class")
    async def client_connection(self):
        proc, client_connection, client_sync = qscope.server.start_bg(
            "mock",
            # server_log_path="./qscope.server.log",
            log_level=TEST_LOGLEVEL,
        )
        yield client_connection
        qscope.server.close_bg(proc, client_connection)

    @pytest_asyncio.fixture(loop_scope="class", scope="class")
    async def notif_listener(self, client_connection):
        logger.info("Starting notif listener.")
        tsk, qu = qscope.server.start_bg_notif_listener(client_connection)
        await asyncio.sleep(0.1)
        logger.info("Started notif listener.")

        class notif_listener:
            def __init__(self, tsk, qu):
                self.tsk = tsk
                self.qu = qu

        nl = notif_listener(tsk=tsk, qu=qu)
        yield nl
        nl.tsk.cancel()

    @pytest.mark.asyncio(loop_scope="class")
    async def test_notif_new_meas(self, client_connection, notif_listener):
        config = TESTING_MEAS_CONFIG
        await asyncio.sleep(0.01)
        meas_id = qscope.server.add_measurement(client_connection, config)

        await wait_for_notif(notif_listener.qu, NewMeasurement)

        qscope.server.stop_measurement(client_connection, meas_id)
        qscope.server.close_measurement_wait(client_connection, meas_id)

        notif = await wait_for_notif(notif_listener.qu, MeasurementUpdate)
        assert notif.meas_id == meas_id
        assert notif.new_state == MEAS_STATE.FINISHED
        qscope.server.clean_queue(notif_listener.qu)

    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.slow
    async def test_notif_stream(self, client_connection, notif_listener):
        qscope.server.startup(client_connection)
        qscope.server.camera_start_video(client_connection)
        await wait_for_notif(notif_listener.qu, NewStream)
        qscope.server.camera_stop_video(client_connection)
        await asyncio.sleep(0.1)
        qscope.server.packdown(client_connection)
        qscope.server.clean_queue(notif_listener.qu)

    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.slow
    async def test_notif_measupdates(self, client_connection, notif_listener):
        config = TESTING_MEAS_CONFIG
        meas_id = qscope.server.add_measurement(client_connection, config)
        await wait_for_notif(notif_listener.qu, NewMeasurement)
        qscope.server.start_measurement_wait(client_connection, meas_id)
        await wait_for_notif(notif_listener.qu, MeasurementUpdate)
        await asyncio.sleep(0.1)
        qscope.server.measurement_set_rolling_avg_window(client_connection, meas_id, 2)
        await asyncio.sleep(0.1)
        await wait_for_notif(notif_listener.qu, MeasurementFrame)
        # below may take a while... (4s on my machine)
        await wait_for_notif(notif_listener.qu, RollingAvgSweepUpdate, timeout=12)
        qscope.server.stop_measurement(client_connection, meas_id)
        qscope.server.close_measurement_wait(client_connection, meas_id)
        qscope.server.clean_queue(notif_listener.qu)

    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.slow
    async def test_notif_changeaoiframenum(self, client_connection, notif_listener):
        config = TESTING_MEAS_CONFIG
        meas_id = qscope.server.add_measurement(client_connection, config)
        qscope.server.start_measurement_wait(client_connection, meas_id)

        prev_sweep_resp = await wait_for_notif(notif_listener.qu, SweepUpdate)
        prev_frame_resp = await wait_for_notif(notif_listener.qu, MeasurementFrame)

        qscope.server.measurement_set_aoi(client_connection, meas_id, (10, 10, 20, 20))
        qscope.server.measurement_set_frame_num(client_connection, meas_id, 5)
        await asyncio.sleep(0.1)
        qscope.server.clean_queue(notif_listener.qu)

        new_sweep_resp = await wait_for_notif(notif_listener.qu, SweepUpdate)
        new_frame_resp = await wait_for_notif(
            notif_listener.qu, MeasurementFrame, timeout=2
        )

        assert not np.all(new_sweep_resp.sweep_data == prev_sweep_resp.sweep_data)
        assert not np.all(new_sweep_resp.aoi == prev_sweep_resp.aoi)
        assert not np.all(new_frame_resp.sig_frame == prev_frame_resp.sig_frame)

        qscope.server.stop_measurement(client_connection, meas_id)
        qscope.server.close_measurement_wait(client_connection, meas_id)
        qscope.server.clean_queue(notif_listener.qu)
