from unittest.mock import MagicMock, patch

import click.testing
import pytest

from qscope.cli import cli
from qscope.util import DEFAULT_HOST_ADDR, DEFAULT_LOGLEVEL, DEFAULT_PORT


@pytest.fixture
def cli_runner():
    return click.testing.CliRunner()


class TestGUICLI:
    @patch("qscope.gui.main_gui.QApplication")
    @patch("qscope.gui.main_gui.MainWindow")
    def test_default_values(self, mock_window, mock_app, cli_runner):
        # Configure mock app to return 0 from exec()
        mock_app.return_value.exec.return_value = 0

        result = cli_runner.invoke(cli, ["gui"])
        assert result.exit_code == 0
        mock_window.assert_called_once()
        mock_app.assert_called_once()

    @patch("qscope.gui.main_gui.QApplication")
    @patch("qscope.gui.main_gui.MainWindow")
    def test_all_arguments(self, mock_window, mock_app, cli_runner):
        # Configure mock app to return 0 from exec()
        mock_app.return_value.exec.return_value = 0
        result = cli_runner.invoke(
            cli,
            [
                "gui",
                "--system-name",
                "mock",
                "--host-address",
                "localhost",
                "--msg-port",
                "5555",
                "--no-log-to-file",
                "--no-log-to-stdout",
                "--log-path",
                "/tmp/test.log",
                "--clear-prev-log",
                "--log-level",
                "DEBUG",
                "--no-auto-connect",
            ],
        )
        assert result.exit_code == 0
        mock_window.assert_called_once()
        mock_app.assert_called_once()

    def test_host_port_validation(self, cli_runner):
        # Test that providing host without port fails
        result = cli_runner.invoke(cli, ["gui", "--host-address", "localhost"])
        assert result.exit_code != 0
        assert "Must define both" in result.output


class TestServerCLI:
    @patch("qscope.cli.base.start_server")
    def test_default_values(self, mock_start_server, cli_runner):
        result = cli_runner.invoke(cli, ["server", "--system-name", "mock"])
        assert result.exit_code == 0
        mock_start_server.assert_called_once()

    @patch("qscope.cli.base.start_server")
    def test_all_arguments(self, mock_start_server, cli_runner):
        result = cli_runner.invoke(
            cli,
            [
                "server",
                "--system-name",
                "mock",
                "--host-address",
                "localhost",
                "--msg-port",
                "5555",
                "--notif-port",
                "5556",
                "--stream-port",
                "5557",
                "--log-to-file",
                "--log-to-stdout",
                "--log-path",
                "/tmp/server.log",
                "--clear-prev-log",
                "--log-level",
                "DEBUG",
            ],
        )
        assert result.exit_code == 0
        mock_start_server.assert_called_once()


class TestListCommand:
    def test_list_command(self, cli_runner):
        result = cli_runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        # Basic check that it runs - actual server listing is tested in test_bg_killer.py


class TestKillCommand:
    def test_kill_command(self, cli_runner):
        result = cli_runner.invoke(cli, ["kill"])
        assert result.exit_code == 0
        # Basic check that it runs - actual server killing is tested in test_bg_killer.py
