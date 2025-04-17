import asyncio
import os
import time

import numpy as np
import pytest
import pytest_asyncio
from loguru import logger

import qscope
import qscope.meas
import qscope.server
import qscope.util
from qscope.meas import MockSGAndorESRConfig
from qscope.server.client import wait_for_notif, clean_queue
from qscope.types import TESTING_MEAS_CONFIG
from qscope.util import TEST_LOGLEVEL


class TestServerMeas:
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
    @pytest.mark.slow
    async def test_meas_comms(self, client_connection, notif_listener):
        config = TESTING_MEAS_CONFIG

        meas_info = qscope.server.get_all_meas_info(client_connection)
        assert meas_info == dict()

        meas_id = qscope.server.add_measurement(client_connection, config)

        meas_info = qscope.server.get_all_meas_info(client_connection)
        assert meas_id in meas_info

        num_sweeps = meas_info[meas_id]["nsweeps"]

        qscope.server.start_measurement_wait(client_connection, meas_id)
        await asyncio.sleep(0.1)
        qscope.server.clean_queue(notif_listener.qu)

        await wait_for_notif(
            notif_listener.qu, qscope.types.MeasurementFrame, timeout=12
        )
        await wait_for_notif(
            notif_listener.qu, qscope.types.MeasurementFrame, timeout=5
        )

        meas_info2 = qscope.server.measurement_get_info(client_connection, meas_id)
        assert meas_info2["nsweeps"] > num_sweeps

        state = qscope.server.measurement_get_state(client_connection, meas_id)

        sweep_data = qscope.server.measurement_get_sweep(client_connection, meas_id)

        single_frame = qscope.server.measurement_get_frame(
            client_connection, meas_id, "ref", 2
        )

        qscope.server.pause_endsweep_measurement(client_connection, meas_id)
        meas_info = qscope.server.measurement_get_info(client_connection, meas_id)

        qscope.server.start_measurement_wait(client_connection, meas_id)

        qscope.server.measurement_set_frame_num(client_connection, meas_id, 2)
        qscope.server.measurement_set_aoi(client_connection, meas_id, (0, 0, 10, 10))

        qscope.server.stop_measurement(client_connection, meas_id)

        meas_info_before = qscope.server.get_all_meas_info(client_connection)
        qscope.server.close_measurement_wait(client_connection, meas_id)
        meas_info_after = qscope.server.get_all_meas_info(client_connection)
        assert meas_info_before != meas_info_after

        meas_info = qscope.server.get_all_meas_info(client_connection)
        assert meas_info == dict()

    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.slow
    async def test_multimeas(self, client_connection, notif_listener):
        config = TESTING_MEAS_CONFIG

        meas_info = qscope.server.get_all_meas_info(client_connection)
        assert meas_info == dict()

        meas1 = qscope.server.add_measurement(client_connection, config)

        meas_info = qscope.server.get_all_meas_info(client_connection)
        assert meas1 in meas_info

        meas2 = qscope.server.add_measurement(client_connection, config)

        meas_info = qscope.server.get_all_meas_info(client_connection)
        assert meas2 in meas_info

        await asyncio.sleep(0.5)
        qscope.server.clean_queue(notif_listener.qu)

        qscope.server.start_measurement_wait(client_connection, meas1)
        await wait_for_notif(notif_listener.qu, qscope.types.SweepUpdate, timeout=5)

        qscope.server.pause_endsweep_measurement(client_connection, meas1)

        qscope.server.start_measurement_wait(client_connection, meas2)
        qscope.server.measurement_set_frame_num(client_connection, meas2, 2)
        qscope.server.measurement_set_aoi(client_connection, meas2, (0, 0, 10, 10))
        await wait_for_notif(notif_listener.qu, qscope.types.SweepUpdate)

        qscope.server.stop_measurement(client_connection, meas1)
        qscope.server.stop_measurement(client_connection, meas2)

        qscope.server.close_measurement_wait(client_connection, meas1)
        qscope.server.close_measurement_wait(client_connection, meas2)

    # noinspection PyArgumentList
    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.slow
    async def test_meas_avgperpoint(self, client_connection, notif_listener):
        config_dict = TESTING_MEAS_CONFIG.to_dict()
        config_dict["avg_per_point"] = 2
        config = MockSGAndorESRConfig.from_dict(config_dict)

        meas_info = qscope.server.get_all_meas_info(client_connection)
        assert meas_info == dict()

        meas_id = qscope.server.add_measurement(client_connection, config)

        meas_info = qscope.server.get_all_meas_info(client_connection)
        assert meas_id in meas_info

        qscope.server.start_measurement_wait(client_connection, meas_id)
        await wait_for_notif(notif_listener.qu, qscope.types.MeasurementUpdate)

        qscope.server.clean_queue(notif_listener.qu)
        qscope.server.pause_endsweep_measurement(client_connection, meas_id)
        await wait_for_notif(notif_listener.qu, qscope.types.MeasurementUpdate)

        qscope.server.start_measurement_wait(client_connection, meas_id, timeout=5)

        qscope.server.measurement_set_frame_num(client_connection, meas_id, 2)
        qscope.server.measurement_set_aoi(client_connection, meas_id, (0, 0, 10, 10))

        qscope.server.stop_measurement(client_connection, meas_id)

        qscope.server.close_measurement_wait(client_connection, meas_id)

    # noinspection PyArgumentList
    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.slow
    async def test_meas_saving(self, client_connection, notif_listener):
        config = TESTING_MEAS_CONFIG
        meas_info = qscope.server.get_all_meas_info(client_connection)
        assert meas_info == dict()

        meas_id = qscope.server.add_measurement(client_connection, config)

        qscope.server.clean_queue(notif_listener.qu)
        qscope.server.start_measurement_wait(client_connection, meas_id)
        await asyncio.sleep(5)
        qscope.server.pause_endsweep_measurement(client_connection, meas_id)
        await wait_for_notif(notif_listener.qu, qscope.types.MeasurementUpdate)

        # data is: np.vstack((self.sweep_x, self.sweep_y_sig, self.sweep_y_ref))
        # i.e. has shape: (3, nsweeps)
        data = qscope.server.measurement_get_sweep(client_connection, meas_id)
        sweep_path = qscope.server.measurement_save_sweep(
            client_connection, meas_id, "test"
        )
        fd_path = qscope.server.measurement_save_full_data(
            client_connection, meas_id, "test"
        )
        saved_sweep_data = np.load(sweep_path)
        assert saved_sweep_data.shape == data.shape
        assert np.allclose(saved_sweep_data, data)

        notif = await wait_for_notif(notif_listener.qu, qscope.types.SaveFullComplete)
        fd_path = notif.save_path
        saved_fd_data = np.load(fd_path)  # NPz object
        saved_sig_data = saved_fd_data["y_sig"]
        # saved_ref_data = saved_fd_data["y_ref"]
        frame_shape = qscope.server.measurement_get_frame(
            client_connection, meas_id
        ).shape
        assert saved_sig_data.shape == (data.shape[-1], *frame_shape)

        qscope.server.stop_measurement(client_connection, meas_id)
        qscope.server.close_measurement_wait(client_connection, meas_id)
        
    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.slow
    async def test_meas_stop_detection(self, client_connection, notif_listener):
        """Test that measurement stop detection works correctly in scripting functions."""
        from qscope.types import MeasurementStoppedError, SweepUpdate, MeasurementUpdate
        import threading
        import warnings
        
        # Create a connection manager for scripting functions
        manager = qscope.server.ConnectionManager()
        manager._connection = client_connection
        manager._client_sync = qscope.server.client.client_sync(client_connection)
        manager._notif_queue = notif_listener.qu
        
        # Add and start a measurement
        config = TESTING_MEAS_CONFIG
        meas_id = qscope.server.add_measurement(client_connection, config)
        qscope.server.start_measurement_wait(client_connection, meas_id)
        
        # Clean the queue to start fresh
        qscope.server.clean_queue(notif_listener.qu)
        
        # Wait for at least one SweepUpdate to arrive
        sweep_update = await wait_for_notif(notif_listener.qu, SweepUpdate, timeout=5)
        assert sweep_update.meas_id == meas_id
        
        # Now stop the measurement
        qscope.server.stop_measurement(client_connection, meas_id)
        
        # Wait for the measurement to be marked as stopped
        update = await wait_for_notif(notif_listener.qu, MeasurementUpdate, timeout=5)
        assert update.meas_id == meas_id
        assert update.new_state == "FINISHED"
        
        # Now test the wait_for_notification_with_meas_check function
        # Add a SweepUpdate to the queue
        fake_sweep_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        notif_listener.qu.put_nowait(
            SweepUpdate(
                meas_id=meas_id,
                sweep_progress=50.0,
                nsweeps=1,
                aoi=None,
                sweep_data=fake_sweep_data
            )
        )
        
        # Add a MeasurementUpdate to the queue
        notif_listener.qu.put_nowait(
            MeasurementUpdate(
                meas_id=meas_id,
                old_state="RUNNING",
                new_state="FINISHED"
            )
        )
        
        # Test with raise_on_stop=True
        try:
            # First, make sure the queue is empty
            clean_queue(notif_listener.qu)
            
            # Add a SweepUpdate to the queue first (this will be stored as latest_matching_notif)
            notif_listener.qu.put_nowait(
                SweepUpdate(
                    meas_id=meas_id,
                    sweep_progress=50.0,
                    nsweeps=1,
                    aoi=None,
                    sweep_data=fake_sweep_data
                )
            )
            
            # Then add a MeasurementUpdate with FINISHED state
            notif_listener.qu.put_nowait(
                MeasurementUpdate(
                    meas_id=meas_id,
                    old_state="RUNNING",
                    new_state="FINISHED"
                )
            )
            
            # Now try to wait for a SweepUpdate - this should raise MeasurementStoppedError
            # with the latest SweepUpdate we just added
            await manager.wait_for_notification_with_meas_check(
                SweepUpdate, meas_id, timeout=1
            )
            pytest.fail("Expected MeasurementStoppedError was not raised")
        except MeasurementStoppedError as e:
            # Verify that the exception contains the latest notification
            assert e.latest_notification is not None
            assert hasattr(e.latest_notification, 'sweep_data')
            assert np.array_equal(e.latest_notification.sweep_data, fake_sweep_data)
        
        # Clean up
        qscope.server.close_measurement_wait(client_connection, meas_id)


@pytest.mark.asyncio(loop_scope="class")
class TestSaveNotes:
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
            log_level=TEST_LOGLEVEL,
        )
        qscope.server.startup(client_connection)
        yield client_connection
        qscope.server.packdown(client_connection)
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
    async def test_save_sweep_notes(self, client_connection, notif_listener):
        """Test saving notes with sweep data"""
        config = TESTING_MEAS_CONFIG
        meas_id = qscope.server.add_measurement(client_connection, config)

        # Run measurement briefly
        qscope.server.start_measurement_wait(client_connection, meas_id)
        await asyncio.sleep(0.5)
        qscope.server.pause_endsweep_measurement(client_connection, meas_id)

        # Test saving with notes
        test_notes = "Test measurement notes\nMultiple lines"
        save_path = qscope.server.measurement_save_sweep(
            client_connection, meas_id, "test_notes", notes=test_notes
        )

        # Verify notes file exists and contains correct content
        notes_path = save_path[:-10] + ".md"
        assert os.path.exists(notes_path)
        with open(notes_path, "r") as f:
            saved_notes = f.read()
        assert saved_notes == test_notes

        qscope.server.close_measurement_wait(client_connection, meas_id)

    @pytest.mark.asyncio(loop_scope="class")
    async def test_save_snapshot_notes(self, client_connection):
        """Test saving notes with snapshot"""
        test_notes = "Snapshot test notes"
        save_path = qscope.server.camera_take_and_save_snapshot(
            client_connection, "test_notes", notes=test_notes
        )

        notes_path = save_path.rsplit(".", 1)[0] + ".md"
        assert os.path.exists(notes_path)
        with open(notes_path, "r") as f:
            saved_notes = f.read()
        assert saved_notes == test_notes

    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.slow
    async def test_save_stream_notes(self, client_connection):
        """Test saving notes with stream data"""
        # Start video stream
        qscope.server.camera_start_video(client_connection)
        await asyncio.sleep(0.5)  # Wait for stream to start

        test_notes = "Stream test notes"
        save_path = qscope.server.save_latest_stream(
            client_connection, "test_notes", color_map="seaborn:mako", notes=test_notes
        )

        notes_path = save_path.rsplit(".", 1)[0] + ".md"
        assert os.path.exists(notes_path)
        with open(notes_path, "r") as f:
            saved_notes = f.read()
        assert saved_notes == test_notes

        qscope.server.camera_stop_video(client_connection)

    @pytest.mark.asyncio(loop_scope="class")
    @pytest.mark.slow
    async def test_save_full_data_notes(self, client_connection, notif_listener):
        """Test saving notes with full measurement data"""
        config = TESTING_MEAS_CONFIG
        meas_id = qscope.server.add_measurement(client_connection, config)

        # Run measurement briefly
        qscope.server.start_measurement_wait(client_connection, meas_id)
        await asyncio.sleep(0.5)
        qscope.server.pause_endsweep_measurement(client_connection, meas_id)

        test_notes = "Full data test notes"
        qscope.server.measurement_save_full_data(
            client_connection, meas_id, "test_notes", notes=test_notes
        )

        # Wait for save completion notification
        notif = await wait_for_notif(notif_listener.qu, qscope.types.SaveFullComplete)
        save_path = notif.save_path

        notes_path = save_path[:-4] + ".md"
        assert os.path.exists(notes_path)
        with open(notes_path, "r") as f:
            saved_notes = f.read()
        assert saved_notes == test_notes

        qscope.server.close_measurement_wait(client_connection, meas_id)
