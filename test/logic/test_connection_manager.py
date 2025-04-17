"""Tests for ConnectionManager class"""

import asyncio

import pytest
import pytest_asyncio
from loguru import logger

import qscope
from qscope.server.connection_manager import ConnectionManager
from qscope.types import TESTING_MEAS_CONFIG, MeasurementUpdate, NewMeasurement
from qscope.util import TEST_LOGLEVEL


class TestConnectionManager:
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

    @pytest_asyncio.fixture(scope="function")
    async def manager(self):
        manager = ConnectionManager()
        yield manager
        manager.stop_server()
        manager.clean_notification_queue()
        await asyncio.sleep(0.1)  # Allow cleanup to complete

    def test_initial_state(self, manager: ConnectionManager):
        """Test initial state of ConnectionManager"""
        assert not manager.is_connected()
        assert manager.connection is None
        assert manager.client_sync is None

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_server_lifecycle(self, manager: ConnectionManager):
        """Test starting/stopping server"""
        manager.start_local_server("MOCK")
        await asyncio.sleep(2.0)  # Allow server to fully start and initialize

        manager.connect()
        assert manager.is_connected()
        assert manager.connection is not None
        assert manager.client_sync is not None

        manager.disconnect()
        assert not manager.is_connected()
        assert manager.connection is None
        assert manager.client_sync is None

        manager.stop_server()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_notification_handling(self, manager: ConnectionManager):
        """Test notification handling"""
        manager.start_local_server("MOCK")
        await asyncio.sleep(0.1)  # Allow server to fully start and initialize

        manager.connect()
        manager.start_notification_listener()
        manager.startup()

        # Add a measurement which should generate notifications
        meas_id = manager.add_measurement(TESTING_MEAS_CONFIG)

        # Wait for new measurement notification
        notif = await manager.wait_for_notification(NewMeasurement, timeout=5)
        assert isinstance(notif, NewMeasurement)
        assert notif.meas_id == meas_id

        # Clean up
        manager.stop_measurement(meas_id)
        manager.close_measurement_wait(meas_id)

        # Wait for measurement update notification
        notif = await manager.wait_for_notification(MeasurementUpdate, timeout=5)
        assert isinstance(notif, MeasurementUpdate)
        assert notif.meas_id == meas_id

        manager.clean_notification_queue()
        manager.packdown()
        manager.stop_server()
        manager.disconnect()

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_method_delegation(self, manager: ConnectionManager):
        """Test delegation to client module methods"""
        manager.start_local_server("MOCK")
        await asyncio.sleep(0.1)

        manager.connect()

        # Test some delegated methods
        assert manager.echo("test") == "test"
        assert manager.ping() == "pong"

        manager.stop_server()
        manager.disconnect()

    def test_error_handling(self, manager):
        """Test error handling for invalid operations"""
        with pytest.raises(RuntimeError):
            manager.echo("test")  # Should fail when not connected

        with pytest.raises(RuntimeError):
            manager.start_notification_listener()  # Should fail when not connected

        with pytest.raises(RuntimeError):
            manager.get_stream_socket()  # Should fail when not connected
