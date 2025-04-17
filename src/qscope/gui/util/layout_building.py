# Version: 1.0
# Date: 2021-07-20
from typing import Iterator, Sequence, Type

from PyQt6.QtCore import *

# Functions for handling GUI errors
# Import necessary modules
from PyQt6.QtWidgets import *

from qscope.gui.widgets import QuComboBox


def add_row_to_layout(layout, *widgets: QWidget | tuple[QWidget, int]):
    """Helper method to add a row of widgets to a layout."""
    row = QHBoxLayout()
    for widget in widgets:
        if isinstance(widget, tuple):
            widget, width = widget
            row.addWidget(widget, width)
        else:
            row.addWidget(widget)
    layout.addLayout(row)


# frame_hbox_widgets
def frame_hbox_widgets(widgets):
    frame = QFrame()
    # frame.setFrameStyle(QFrame.Shape.StyledPanel)
    layout = QHBoxLayout(frame)
    for widget in widgets:
        layout.addWidget(widget)
    layout.setSpacing(0)
    layout.setContentsMargins(0, 0, 0, 0)
    return frame


def remove_layout_margin(layout):
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    return layout


def get_spin_box_wdiget(
    label: str,
    default_value: int,
    min_value: int = 0,
    max_value: int = 1000000000,
    single_step: int = 1,
    decimals: int = 0,
):
    label_widget = QLabel(label)
    if decimals == 0:
        spin_box = QSpinBox()
    else:
        spin_box = QDoubleSpinBox()
    spin_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
    spin_box.setRange(min_value, max_value)
    spin_box.setValue(default_value)
    spin_box.setSingleStep(single_step)
    if decimals > 0:
        spin_box.setDecimals(decimals)
    return label_widget, spin_box


def get_dropdown_widget(label: str, items: list[str]):
    label_widget = QLabel(label)
    dropdown = QuComboBox()
    # dropdown.setEditable(True)
    # dropdown.lineEdit().setReadOnly(True)
    # dropdown.lineEdit().setAlignment(Qt.AlignmentFlag.AlignCenter)
    for item in items:
        dropdown.addItem(item)
    return label_widget, dropdown
