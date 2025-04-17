from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, Optional, TypeVar

from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QWidget

from .combo_boxes import QuComboBox
from .config import WidgetConfig
from .spin_boxes import QuDoubleSpinBox, QuSpinBox
from .types import WidgetType


def create_widget_from_config(config: WidgetConfig) -> QWidget:
    """Create a widget based on configuration"""
    if config.widget_type == WidgetType.INT:
        widget = QuSpinBox(config)
        widget.setRange(config.min_value, config.max_value)
        widget.setValue(config.default)

        widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
    elif config.widget_type == WidgetType.FLOAT:
        widget = QuDoubleSpinBox(config)
        widget.setRange(config.min_value, config.max_value)
        widget.setValue(config.default)

        widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if config.transform:
            widget.valueChanged.connect(lambda x: widget.setValue(config.transform(x)))
    # elif config.widget_type == WidgetType.SCI:
    #     widget = QuScientificSpinBox(config)
    #     widget.setRange(config.min_value, config.max_value)
    #     widget.setValue(config.default)
    #
    #     widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
    #     if config.transform:
    #         widget.valueChanged.connect(lambda x: widget.setValue(config.transform(x)))

    elif config.widget_type == WidgetType.CHOICE:
        widget = QuComboBox(config)
        # widget.setEditable(True)
        # widget.lineEdit().setReadOnly(True)
        # widget.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
        widget.addItems(config.choices)
        idx = widget.findText(str(config.default))
        if idx >= 0:
            widget.setCurrentIndex(idx)
    else:
        raise ValueError(f"Unknown widget type: {config.widget_type}")

    return widget
