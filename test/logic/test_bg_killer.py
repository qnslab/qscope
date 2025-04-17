import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil
import pytest

from qscope.server.bg_killer import (
    cleanup_stale_servers,
    get_servers_dir,
    kill_qscope_servers,
    list_running_servers,
)


@pytest.fixture
def mock_servers_dir(tmp_path):
    """Create a temporary directory for test PID files."""
    servers_dir = tmp_path / ".qscope" / "running_servers"
    servers_dir.mkdir(parents=True)
    with patch("qscope.server.bg_killer.get_servers_dir") as mock_get_dir:
        mock_get_dir.return_value = servers_dir
        yield servers_dir


def create_test_pid_file(
    servers_dir: Path, pid: int, timestamp: str = "2024-01-01_12:00:00"
):
    """Helper to create a test PID file."""
    server_info = {
        "pid": pid,
        "timestamp": timestamp,
        "host": "localhost",
        "ports": {"msg": 5555, "notif": 5556, "stream": 5557},
    }
    pid_file = servers_dir / f"server_{pid}.json"
    with pid_file.open("w") as f:
        json.dump(server_info, f)
    return pid_file


def test_get_servers_dir():
    """Test that get_servers_dir creates and returns correct directory."""
    servers_dir = get_servers_dir()
    assert servers_dir.exists()
    assert servers_dir.is_dir()
    assert str(servers_dir).endswith("running_servers")


def test_list_running_servers_empty(mock_servers_dir):
    """Test listing servers when none exist."""
    servers = list_running_servers()
    assert len(servers) == 0


def test_list_running_servers_with_active(mock_servers_dir):
    """Test listing servers with an active process."""
    # Use current process as a "running server"
    current_pid = os.getpid()
    create_test_pid_file(mock_servers_dir, current_pid)

    servers = list_running_servers()
    assert len(servers) == 1
    assert servers[0]["pid"] == current_pid
    assert servers[0]["running"] == True


def test_list_running_servers_with_dead(mock_servers_dir):
    """Test listing servers with a non-existent PID."""
    # Use an unlikely to exist PID
    fake_pid = 999999
    create_test_pid_file(mock_servers_dir, fake_pid)

    servers = list_running_servers()
    assert len(servers) == 1
    assert servers[0]["pid"] == fake_pid
    assert servers[0]["running"] == False


def test_kill_qscope_servers_no_servers(mock_servers_dir):
    """Test kill_qscope_servers when no servers exist."""
    killed = kill_qscope_servers()
    assert killed == 0


@patch("psutil.Process")
def test_kill_qscope_servers_with_servers(mock_process, mock_servers_dir):
    """Test killing active server processes."""
    # Setup mock process
    mock_proc = MagicMock()
    mock_process.return_value = mock_proc

    # Create test PID file
    create_test_pid_file(mock_servers_dir, 12345)

    killed = kill_qscope_servers()

    assert killed == 1
    mock_proc.kill.assert_called_once()
    assert not (mock_servers_dir / "server_12345.json").exists()


def test_cleanup_stale_servers(mock_servers_dir):
    """Test cleanup of stale server PID files."""
    # Create a stale PID file
    stale_pid = 999999
    pid_file = create_test_pid_file(mock_servers_dir, stale_pid)

    cleanup_stale_servers()

    assert not pid_file.exists()


def test_cleanup_stale_servers_keeps_active(mock_servers_dir):
    """Test cleanup preserves active server PID files."""
    # Use current process as "active"
    current_pid = os.getpid()
    pid_file = create_test_pid_file(mock_servers_dir, current_pid)

    cleanup_stale_servers()

    assert pid_file.exists()
