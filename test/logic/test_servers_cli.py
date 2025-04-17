from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from qscope.cli import cli


def test_list_command_no_servers():
    """Test list command when no servers are running."""
    runner = CliRunner()
    with patch("qscope.cli.base.list_running_servers", return_value=[]):
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "No servers found" in result.output


def test_list_command_with_servers():
    """Test list command with running servers."""
    mock_servers = [
        {
            "pid": 12345,
            "timestamp": "2024-01-01_12:00:00",
            "host": "localhost",
            "ports": {"msg": 5555, "notif": 5556, "stream": 5557},
            "running": True,
        }
    ]

    runner = CliRunner()
    # Mock the server list before invoking CLI
    with patch("qscope.cli.base.list_running_servers") as mock_list:
        mock_list.return_value = mock_servers
        result = runner.invoke(cli, ["list"])

        # Verify the mock was called
        mock_list.assert_called_once()

        # Verify exit code and basic output
        assert result.exit_code == 0
        assert "Running qscope servers:" in result.output

        # Verify all mock data appears in output
        assert "PID: 12345 (RUNNING)" in result.output
        assert "Started: 2024-01-01_12:00:00" in result.output
        assert "Host: localhost" in result.output
        assert "Ports: msg=5555, notif=5556, stream=5557" in result.output


def test_kill_command_no_servers():
    """Test kill command when no servers to kill."""
    runner = CliRunner()
    with patch("qscope.cli.base.kill_qscope_servers", return_value=0):
        result = runner.invoke(cli, ["kill"])
        assert result.exit_code == 0
        assert "No running qscope servers found" in result.output


def test_kill_command_with_servers():
    """Test kill command with servers to kill."""
    runner = CliRunner()
    with patch("qscope.cli.base.kill_qscope_servers", return_value=2):
        result = runner.invoke(cli, ["kill"])
        assert result.exit_code == 0
        assert "Killed 2 qscope server(s)" in result.output
