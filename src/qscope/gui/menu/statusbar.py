from loguru import logger
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QLabel,
    QLineEdit,
    QStatusBar,
)

from ..util.settings import GUISettings
from ..widgets.line_edit import QuLineEdit
from .settings_defs import StatusBarSettings


class QScopeStatusBar(QStatusBar):
    """Custom status bar for QScope application with server and hardware status indicators."""

    LED_G_PATH = "./src/qscope/gui/icons/led_green.svg"
    LED_R_PATH = "./src/qscope/gui/icons/led_red.svg"

    def __init__(self, parent=None):
        """Initialize the status bar.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget, by default None
        """
        super().__init__(parent)
        self._create_widgets()
        self._style()

        self.sbar_settings = GUISettings(self, "STATUS BAR OPTS")
        self.sbar_settings.load_prev_state()

    def _style(self):
        """Apply styling to the status bar."""
        # Uncomment desired styling:
        # self.setStyleSheet("QStatusBar{border-top: 1px outset grey;}")
        # self.setStyleSheet("QStatusBar{border: 1px inset grey; border-radius: 5px; margin: 2px;}")
        pass

    def _create_widgets(self):
        """Create and arrange all status bar widgets."""
        # Project tag widgets
        self.project_tag_label = QLabel("Project tag: ")
        self.project_tag = QuLineEdit(StatusBarSettings.project_name)
        # self.project_tag.setText("")
        self.project_tag.setFixedWidth(200)

        # Server status widgets
        self.server_status = QLabel("Server: Not connected")
        self.server_led = QLabel()
        self.server_led.setPixmap(QIcon(self.LED_R_PATH).pixmap(20, 20))

        # Hardware status widgets
        self.hw_status = QLabel("Hardware: Not initialized")
        self.hw_led = QLabel()
        self.hw_led.setPixmap(QIcon(self.LED_R_PATH).pixmap(20, 20))

        # Add all widgets to status bar
        self.addPermanentWidget(self.project_tag_label)
        self.addPermanentWidget(self.project_tag)
        self.addPermanentWidget(self.server_status)
        self.addPermanentWidget(self.server_led)
        self.addPermanentWidget(self.hw_status)
        self.addPermanentWidget(self.hw_led)

    # TODO change to bool argument
    def set_server_status(self, status: str):
        """Update the server connection status display.

        Parameters
        ----------
        status : str
            Status string to display, e.g. "Connected" or "Disconnected"
        """
        self.server_status.setText(f"Server: {status}")
        icon_path = self.LED_G_PATH if status == "Connected" else self.LED_R_PATH
        self.server_led.setPixmap(QIcon(icon_path).pixmap(20, 20))

    # TODO change to bool argument
    def set_hw_status(self, status: str):
        """Update the hardware initialization status display.

        Parameters
        ----------
        status : str
            Status string to display, e.g. "Initialized" or "Not initialized"
        """
        self.hw_status.setText(f"Hardware: {status}")
        icon_path = self.LED_G_PATH if status == "Initialized" else self.LED_R_PATH
        self.hw_led.setPixmap(QIcon(icon_path).pixmap(20, 20))

    def get_project_tag(self) -> str:
        """Get the current project tag.

        Returns
        -------
        str
            Current project tag text
        """
        return self.project_tag.text()
