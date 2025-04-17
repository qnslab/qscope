from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Generic, Optional, TypeVar

from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QLineEdit, QSpinBox, QWidget

from .types import WidgetType

T = TypeVar("T")


@dataclass(kw_only=True)
class WidgetConfig(Generic[T]):
    """Configuration for a GUI widget with type information and validation"""

    name: str
    widget_type: WidgetType
    default: T
    validator: Optional[Callable[[T], bool]] = None
    transform: Optional[Callable[[T], T]] = None
    min_value: Optional[T] = None
    max_value: Optional[T] = None
    choices: Optional[list[str]] = None

    def validate(self, value: T) -> bool:
        """Validate a value against constraints"""
        try:
            if self.min_value is not None and value < self.min_value:
                return False
            if self.max_value is not None and value > self.max_value:
                return False
            if self.choices is not None and str(value) not in self.choices:
                return False
            if self.validator is not None and not self.validator(value):
                return False
            return True
        except Exception as e:
            logger.error(f"Validation error for {self.name}: {e}")
            return False

    def get_value(self, widget: QSpinBox | QDoubleSpinBox | QComboBox | QLineEdit) -> T:
        """Get typed value from widget with validation"""
        if self.widget_type == WidgetType.INT:
            value = int(widget.value())
        elif self.widget_type == WidgetType.FLOAT:
            value = float(widget.value())
        elif self.widget_type == WidgetType.STRING:
            value = widget.text()
        elif self.widget_type == WidgetType.CHOICE:
            value = widget.currentText()
            if not value:  # Handle empty text case
                logger.warning(
                    f"Empty combo value for {self.name}, using default {self.default}"
                )
                return self.default
            if value in self.choices:
                return value
            logger.warning(
                f"Invalid combo value '{value}' for {self.name}, using default {self.default}"
            )
            return self.default
        else:
            raise ValueError(f"Unknown widget type: {self.widget_type}")

        # Apply any transform
        if self.transform is not None:
            value = self.transform(value)

        # Validate
        if not self.validate(value):
            logger.warning(
                f"Invalid value {value} for {self.name}, using default {self.default}"
            )
            return self.default

        return value

    def set_value(
        self, widget: QSpinBox | QDoubleSpinBox | QComboBox | QLineEdit, value: T
    ) -> None:
        """Set typed value to widget with validation"""

        if self.widget_type == WidgetType.INT:
            value = int(value)
        elif self.widget_type == WidgetType.FLOAT:
            value = float(value)
        elif self.widget_type == WidgetType.STRING:
            widget.setText(value)
            return

        # Apply any transform
        if self.transform is not None:
            value = self.transform(value)

        # Validate
        if not self.validate(value):
            logger.warning(
                f"Invalid value {value} for {self.name}, using default {self.default}"
            )
            value = self.default

        if self.widget_type == WidgetType.INT:
            widget.setValue(value)
        elif self.widget_type == WidgetType.FLOAT:
            widget.setValue(value)
        elif self.widget_type == WidgetType.CHOICE:
            value_str = str(value)
            if value_str in self.choices:
                index = widget.findText(value_str)
                if index >= 0:
                    widget.setCurrentIndex(index)
                else:
                    logger.error(
                        f"Value '{value_str}' in choices but not in widget items for {self.name}"
                    )
            else:
                logger.warning(
                    f"Invalid combo value '{value_str}' for {self.name}, using default {self.default}"
                )
                index = widget.findText(str(self.default))
                if index >= 0:
                    widget.setCurrentIndex(index)
