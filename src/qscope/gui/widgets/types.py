from enum import Enum
from typing import Any, Callable, Generic, Optional, TypeVar

from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QWidget

T = TypeVar("T")


class WidgetType(Enum):
    INT = "int"
    FLOAT = "float"
    SCI = "sci"
    STRING = "str"
    CHOICE = "choice"
