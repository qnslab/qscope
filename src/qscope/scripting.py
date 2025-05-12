"""Utils for scripting"""

import asyncio
import time
import warnings
from typing import Callable, TypeVar, Any

from loguru import logger

import qscope.server
import qscope.types
from qscope.types import MeasurementStoppedError

T = TypeVar('T')

async def _common_meas_runner(
    manager: qscope.server.ConnectionManager,
    meas_id: str,
    nsweeps: int,
    timeout: float,
    raise_on_stop: bool,
    on_success: Callable[[qscope.types.SweepUpdate], Any]
) -> Any:
    """Common runner function for all measurement wait operations.
    
    Parameters
    ----------
    manager : ConnectionManager
        The connection manager
    meas_id : str
        The measurement ID to monitor
    nsweeps : int
        Number of sweeps to wait for
    timeout : float
        Maximum time to wait
    raise_on_stop : bool
        Whether to raise an exception if the measurement is stopped
    on_success : Callable
        Function to call when the target number of sweeps is reached
        
    Returns
    -------
    Any
        The result of the on_success function
        
    Raises
    ------
    MeasurementStoppedError
        If the measurement is stopped while waiting and raise_on_stop is True,
        or if no data is available
    TimeoutError
        If timeout is reached before the required sweeps are completed
    """
    manager.start_notification_listener()
    start = time.time()
    while time.time() - start < timeout:
        try:
            sweep_update = await manager.wait_for_notification_with_meas_check(
                qscope.types.SweepUpdate, meas_id, timeout=60
            )
            
            if sweep_update.meas_id == meas_id and sweep_update.nsweeps >= nsweeps:
                return on_success(sweep_update)
        except MeasurementStoppedError as e:
            if e.latest_notification is not None:
                if raise_on_stop:
                    raise
                warnings.warn(f"Measurement stopped: {e}. Returning latest data.")
                return e.latest_notification.sweep_data
            else:
                # No data available, always raise an error
                raise MeasurementStoppedError(
                    f"Measurement stopped and no data available: {e}"
                )
    else:
        raise TimeoutError("Timeout waiting for nsweeps")


def meas_wait_for_nsweeps(
    manager: qscope.server.ConnectionManager, 
    meas_id: str,
    nsweeps: int, 
    timeout: float = 10,
    raise_on_stop: bool = False
):
    """Wait for a measurement to complete a specific number of sweeps.
    
    Parameters
    ----------
    manager : ConnectionManager
        The connection manager
    meas_id : str
        The measurement ID to monitor
    nsweeps : int
        Number of sweeps to wait for
    timeout : float, optional
        Maximum time to wait, by default 10 seconds
    raise_on_stop : bool, optional
        Whether to raise an exception if the measurement is stopped, by default False
        
    Returns
    -------
    np.ndarray
        The sweep data if successful, or the latest data if measurement was stopped
        
    Raises
    ------
    MeasurementStoppedError
        If the measurement is stopped while waiting and raise_on_stop is True,
        or if no data is available
    TimeoutError
        If timeout is reached before the required sweeps are completed
    """
    async def runner():
        return await _common_meas_runner(
            manager, meas_id, nsweeps, timeout, raise_on_stop,
            lambda sweep_update: sweep_update.sweep_data
        )
    
    return asyncio.run(runner())


def meas_stop_after_nsweeps(
    manager: qscope.server.ConnectionManager,
    meas_id: str,
    nsweeps: int,
    timeout: float = 10,
    raise_on_stop: bool = False
):
    """Stop a measurement after a specific number of sweeps.
    
    Parameters
    ----------
    manager : ConnectionManager
        The connection manager
    meas_id : str
        The measurement ID to monitor
    nsweeps : int
        Number of sweeps to wait for before stopping
    timeout : float, optional
        Maximum time to wait, by default 10 seconds
    raise_on_stop : bool, optional
        Whether to raise an exception if the measurement is stopped, by default False
        
    Returns
    -------
    np.ndarray
        The sweep data if successful, or the latest data if measurement was stopped
        
    Raises
    ------
    MeasurementStoppedError
        If the measurement is stopped while waiting and raise_on_stop is True,
        or if no data is available
    TimeoutError
        If timeout is reached before the required sweeps are completed
    """
    async def runner():
        return await _common_meas_runner(
            manager, meas_id, nsweeps, timeout, raise_on_stop,
            lambda sweep_update: (manager.stop_measurement(meas_id), sweep_update.sweep_data)[1]
        )
    
    return asyncio.run(runner())


def meas_close_after_nsweeps(
    manager: qscope.server.ConnectionManager,
    meas_id: str,
    nsweeps: int,
    timeout: float = 10,
    raise_on_stop: bool = False
):
    """Stop and close a measurement after a specific number of sweeps.
    
    Parameters
    ----------
    manager : ConnectionManager
        The connection manager
    meas_id : str
        The measurement ID to monitor
    nsweeps : int
        Number of sweeps to wait for before stopping and closing
    timeout : float, optional
        Maximum time to wait, by default 10 seconds
    raise_on_stop : bool, optional
        Whether to raise an exception if the measurement is stopped, by default False
        
    Returns
    -------
    np.ndarray
        The sweep data if successful, or the latest data if measurement was stopped
        
    Raises
    ------
    MeasurementStoppedError
        If the measurement is stopped while waiting and raise_on_stop is True,
        or if no data is available
    TimeoutError
        If timeout is reached before the required sweeps are completed
    """
    async def runner():
        def on_success(sweep_update):
            manager.stop_measurement(meas_id)
            manager.close_measurement_wait(meas_id)
            return sweep_update.sweep_data
            
        return await _common_meas_runner(
            manager, meas_id, nsweeps, timeout, raise_on_stop, on_success
        )
    
    return asyncio.run(runner())
