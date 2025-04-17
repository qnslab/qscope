import PyQt6
from loguru import logger
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QInputDialog, QLabel, QMainWindow, QMenu, QMenuBar

import qscope.server
import qscope.util
from qscope.gui.util import show_warning
from qscope.server.connection_manager import ConnectionManager


# Class for the server options
class ServerMenu(QMenu):
    def __init__(self, parent, menuBar):
        super().__init__()
        self.parent = parent
        self._create_server_menu(menuBar)

    def _create_server_menu(self, menuBar):
        ## Server menu
        server_menu = menuBar.addMenu("&Server")

        # display the server connection status
        self.server_connection_label = QAction("Server not connected")
        self.server_connection_label.setDisabled(True)
        self.server_connection_label.setFont(PyQt6.QtGui.QFont("Arial", 10))

        start_server_action = QAction("&Start server", self.parent)
        start_server_action.triggered.connect(self._start_server)

        server_connect_action = QAction("&Connect to server", self.parent)
        server_connect_action.triggered.connect(self._connect_to_server)

        server_startup_hw_action = QAction("&Startup server hardware", self.parent)
        server_startup_hw_action.triggered.connect(self._server_hw_startup)

        server_check_connection_action = QAction(
            "&Check server connection", self.parent
        )
        server_check_connection_action.triggered.connect(self._check_server_connection)

        server_disconnect_action = QAction("&Disconnect from server", self.parent)
        server_disconnect_action.triggered.connect(self._disconnect_from_server)

        close_server_action = QAction("&Close server", self.parent)
        close_server_action.triggered.connect(self.close_server)

        server_menu.addAction(self.server_connection_label)
        server_menu.addAction(start_server_action)
        server_menu.addAction(server_connect_action)
        server_menu.addAction(server_startup_hw_action)
        server_menu.addAction(server_check_connection_action)
        server_menu.addAction(server_disconnect_action)
        server_menu.addAction(close_server_action)

        self._set_connection_status(status=False)
        return server_menu

    def _start_server(self):
        # Open a dialog to choose the system and system type
        # TODO get the dialog to select the correct classes from the possible systems

        def _open_server_dialog():
            dialog = QInputDialog()
            dialog.setWindowTitle("Start Server")
            dialog.setLabelText("Enter the system and system type")
            # Add a dropdown menu for the system and system type
            # TODO generate from user/package config
            # also allow general text input??
            dialog.setComboBoxItems(
                [
                    "MOCK",
                    "HQDM",
                    "GMX",
                ]
            )

            dialog.setOkButtonText("Start")
            dialog.setCancelButtonText("Cancel")
            dialog.setTextValue("mock:pulsed_camera")
            dialog.resize(300, 100)
            dialog.exec()
            # if the user cancels the dialog, return
            if not dialog.result():
                return None
            else:
                return dialog.textValue()

        system = _open_server_dialog()
        if system is None:
            return

        system = system.lower()
        try:
            if not hasattr(self.parent, "connection_manager"):
                self.parent.connection_manager = ConnectionManager()

            self.parent.connection_manager.start_local_server(
                system,
                log_to_file=True,
                log_level="DEBUG",
            )
        except Exception as e:
            logger.error(f"Error starting server: {e}")

    def _connect_to_server(self):
        # Open a dialog to get insert the server address and port
        def _open_server_connection_dialog():
            dialog = QInputDialog()
            dialog.setWindowTitle("Connect to Server")
            dialog.setLabelText(
                "Enter the server address and port\n"
                + f"Local server address: 127.0.0.1:{qscope.server.DEFAULT_PORT}\n"
                + f"HQDM server address: 10.203.5.209:{qscope.server.DEFAULT_PORT} \n"
                + f"GMX server address: 10.203.5.178:{qscope.server.DEFAULT_PORT} \n"
            )
            dialog.setInputMode(QInputDialog.InputMode.TextInput)
            dialog.setOkButtonText("Connect")
            dialog.setCancelButtonText("Cancel")
            dialog.setTextValue(
                str(qscope.server.DEFAULT_HOST_ADDR)
                + ":"
                + str(qscope.server.DEFAULT_PORT)
            )
            dialog.resize(300, 100)
            dialog.exec()
            # if the user cancels the dialog, return
            if not dialog.result():
                return None, None
            else:
                return dialog.textValue().split(":")

        server_addr, server_port = _open_server_connection_dialog()
        if server_addr is None:
            return None, None

        # Connect to the server
        try:
            if not hasattr(self.parent, "connection_manager"):
                self.parent.connection_manager = ConnectionManager()

            self.parent.connection_manager.connect(server_addr, int(server_port))
            if self.parent.connection_manager.client_sync.version != qscope.__version__:
                show_warning(
                    "Server and client versions are different: "
                    + f"{self.parent.connection_manager.client_sync.version} vs "
                    f"{qscope.__version__}"
                )
            self.parent.start_notif_listener_thread()
            self.parent.start_video_listener_thread()
            self._set_connection_status(status=True)
            self.parent.statusBar.set_server_status("Connected")
            try:
                self.parent.statusBar.set_hw_status("Initialized" if self.parent.connection_manager.client_sync.hardware_started_up else "Not initialized")
            except:
                pass # idk what good logic is here
        except Exception as e:
            logger.exception("Error connecting to server.")
            self._set_connection_status(status=False)
            show_warning("Error connecting to server: \n" + str(e))
            return None, None

    def _server_hw_startup(self):
        try:
            self.parent.connection_manager.startup()
            self.parent.statusBar.set_hw_status("Initialized")
        except Exception as e:
            logger.exception("Error during gui server startup.")

    def _check_server_connection(self):
        try:
            self.parent.connection_manager.ping()
            self._set_connection_status(status=True)
        except Exception as e:
            self.server_connection_label.setText("Server not connected")
            self._set_connection_status(status=False)
            logger.warning("Server connection check failed.")

    def _disconnect_from_server(self):
        try:
            self.parent.connection_manager.disconnect()
            self._set_connection_status(status=False)
            self.parent.connection_manager._connection = None
            self.parent.statusBar.set_server_status("Not connected")
        except Exception as e:
            logger.exception("Error disconnecting from server.")

    def close_server(self):
        try:
            self.parent.connection_manager.stop_server()
            self._set_connection_status(status=False)
            self.parent.connection_manager._connection = None
            self.parent.statusBar.set_server_status("Not connected")
            self.parent.statusBar.set_hw_status("Not Initialized")
        except Exception as e:
            logger.exception("Error stopping server.")

    def _set_connection_status(self, status=False):
        if status:
            self.server_connection_label.setText("Connected to server")
        else:
            self.server_connection_label.setText("Server not connected")
            
