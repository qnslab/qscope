from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPalette
from PyQt6.QtWidgets import (
    QComboBox,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionComboBox,
    QStylePainter,
)

if TYPE_CHECKING:
    from .config import WidgetConfig


class CenterAlignDelegate(QStyledItemDelegate):
    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.displayAlignment = Qt.AlignmentFlag.AlignCenter


class QuComboBox(QComboBox):
    """Wrapper class for QComboBox to allow for additional attributes
    and for automatic configuration of the widget for saving/loading settings"""

    def __init__(self, config: Optional[WidgetConfig] = None):
        super().__init__()
        delegate = CenterAlignDelegate(self)
        self.setItemDelegate(delegate)
        self.config = config
        if config:
            self.default = config.default

    def paintEvent(self, event):
        painter = QStylePainter(self)
        opt = QStyleOptionComboBox()
        self.initStyleOption(opt)

        painter.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)

        # Draw the text centered
        painter.setPen(self.palette().color(QPalette.ColorRole.Text))
        text_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox,
            opt,
            QStyle.SubControl.SC_ComboBoxEditField,
            self,
        )
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignCenter | Qt.AlignmentFlag.AlignVCenter,
            self.currentText(),
        )

    def set_value(self, value):
        if self.config:
            return self.config.set_value(self, value)

    def get_value(self):
        if self.config:
            return self.config.get_value(self)
        else:
            return self.currentText()
