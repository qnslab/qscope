from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import QLineEdit

if TYPE_CHECKING:
    from .config import WidgetConfig


class QuLineEdit(QLineEdit):
    def __init__(self, config: Optional[WidgetConfig] = None):
        super().__init__()
        self.config = config
        self.default = config.default if config else "data"  # Always set a default

        # Initialize with default value
        if config:
            self.set_value(self.default)
        else:
            self.setText(self.default)

    def set_value(self, value):
        if self.config:
            return self.config.set_value(self, value)
        else:
            self.setText(str(value))

    def get_value(self):
        if self.config:
            return self.config.get_value(self)
        else:
            text = self.text()
            return text if text else self.default  # Return default if empty
