from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

from loguru import logger
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QValidator
from PyQt6.QtWidgets import QComboBox, QDoubleSpinBox, QSpinBox, QWidget

from .types import WidgetType


class SIPrefixMixin:
    """Mixin class providing SI prefix functionality for spin boxes."""

    SI_PREFIXES: Dict[str, float] = {
        "f": 1e-15,
        "p": 1e-12,
        "n": 1e-9,
        "u": 1e-6,
        "m": 1e-3,
        "k": 1e3,
        "M": 1e6,
        "G": 1e9,
        "T": 1e12,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_prefix: Optional[str] = None
        self.setKeyboardTracking(False)
        self.valueChanged.connect(self._update_tooltip)

    def _update_tooltip(self, value):
        """Update tooltip to show the actual value."""
        self.setToolTip(f"Actual value: {value}")

    def get_prefix(self) -> Optional[str]:
        """Get the current SI prefix."""
        return self.current_prefix

    def set_prefix(self, prefix: Optional[str]) -> None:
        """Set the SI prefix to use for display."""
        if prefix is not None and prefix not in self.SI_PREFIXES:
            raise ValueError(f"Invalid SI prefix: {prefix}")
        self.current_prefix = prefix
        self.setValue(self.value())  # Trigger display update


class QuSpinBox(QSpinBox, SIPrefixMixin):
    """Wrapper class for QSpinBox with SI prefix support."""

    INTEGER_SI_PREFIXES = {"k": 1000, "M": 1000000, "G": 1000000000, "T": 1000000000000}

    def __init__(self, WidgetConfig):
        super().__init__()
        self.config = WidgetConfig
        self.default = WidgetConfig.default
        self.SI_PREFIXES = (
            self.INTEGER_SI_PREFIXES
        )  # Override with integer-specific prefixes
        self.REVERSE_PREFIXES = {v: k for k, v in self.SI_PREFIXES.items()}

    def set_value(self, value):
        return self.config.set_value(self, value)

    def get_value(self):
        return self.config.get_value(self)

    def validate(self, text: str, pos: int) -> Tuple[QValidator.State, str, int]:
        """Validate input text."""
        valid_chars = set("0123456789-" + "".join(self.SI_PREFIXES.keys()))

        if text.count("-") > 0 and text[0] != "-":
            return (QValidator.State.Invalid, text, pos)

        if not text:
            return (QValidator.State.Intermediate, text, pos)

        if all(c in valid_chars for c in text):
            try:
                self.valueFromText(text)
                return (QValidator.State.Acceptable, text, pos)
            except ValueError:
                return (QValidator.State.Intermediate, text, pos)

        return (QValidator.State.Invalid, text, pos)

    def valueFromText(self, text: str) -> int:
        """Convert text to value, handling SI prefixes."""
        text = text.strip()
        if not text:
            return 0

        if text[-1] in self.SI_PREFIXES:
            prefix = text[-1]
            number = text[:-1]
            if not number:
                raise ValueError("Invalid number format")
            try:
                self.current_prefix = prefix
                value = int(float(number) * self.SI_PREFIXES[prefix])
                return value
            except ValueError:
                raise ValueError("Invalid number format")

        try:
            self.current_prefix = None
            return int(text)
        except ValueError:
            raise ValueError("Invalid number format")

    def keyPressEvent(self, event):
        """Handle precise value adjustments based on cursor position."""
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            cursor_pos = self.lineEdit().cursorPosition()
            text = self.lineEdit().text()

            # Get the numeric part and any SI prefix
            if text and text[-1] in self.SI_PREFIXES:
                number = text[:-1]
                prefix = text[-1]
            else:
                number = text
                prefix = ""

            # Calculate step size based on cursor position
            power = len(number) - cursor_pos - 1
            if power < 0:
                power = 0
            step = 10**power

            # Get current value
            current_value = int(number)

            # Apply the step
            if event.key() == Qt.Key.Key_Up:
                new_value = current_value + step
            else:
                new_value = current_value - step

            # Apply prefix scaling if needed
            if prefix:
                new_value = new_value * self.SI_PREFIXES[prefix]

            # Set the new value
            self.setValue(new_value)

            # Restore cursor position
            self.lineEdit().setCursorPosition(cursor_pos)
        else:
            super().keyPressEvent(event)

    def stepBy(self, steps: int) -> None:
        """Handle spinbox arrow buttons."""
        text = self.lineEdit().text()

        # Get the numeric part and any SI prefix
        if text and text[-1] in self.SI_PREFIXES:
            number = text[:-1]
            prefix = text[-1]
        else:
            number = text
            prefix = ""

        # Get current value
        current_value = int(number)

        # Apply the step
        new_value = current_value + steps

        # Apply prefix scaling if needed
        if prefix:
            new_value = new_value * self.SI_PREFIXES[prefix]

        # Set the new value
        self.setValue(new_value)

    def textFromValue(self, value: int) -> str:
        """Convert value to text, maintaining SI prefix notation."""
        if value == 0:
            return "0"

        abs_value = abs(value)
        for scale, prefix in sorted(self.REVERSE_PREFIXES.items(), reverse=True):
            if abs_value >= scale and abs_value % (scale // 10) == 0:
                scaled_value = value // scale
                return f"{scaled_value}{prefix}"

        return str(value)


class QuDoubleSpinBox(QDoubleSpinBox, SIPrefixMixin):
    """Wrapper class for QDoubleSpinBox with SI prefix support."""

    def __init__(self, WidgetConfig):
        super().__init__()
        self.config = WidgetConfig
        self.default = WidgetConfig.default
        self.setDecimals(10)  # Allow more decimal places
        self.REVERSE_PREFIXES = {v: k for k, v in self.SI_PREFIXES.items()}

    def set_value(self, value):
        return self.config.set_value(self, value)

    def get_value(self):
        return self.config.get_value(self)

    def validate(self, text: str, pos: int) -> Tuple[QValidator.State, str, int]:
        """Validate input text."""
        valid_chars = set("0123456789.-" + "".join(self.SI_PREFIXES.keys()))

        if text.count(".") > 1 or (text.count("-") > 0 and text[0] != "-"):
            return (QValidator.State.Invalid, text, pos)

        if not text:
            return (QValidator.State.Intermediate, text, pos)

        if all(c in valid_chars for c in text):
            try:
                self.valueFromText(text)
                return (QValidator.State.Acceptable, text, pos)
            except ValueError:
                return (QValidator.State.Intermediate, text, pos)

        return (QValidator.State.Invalid, text, pos)

    def valueFromText(self, text: str) -> float:
        """Convert text to value, handling SI prefixes."""
        text = text.strip()
        if not text:
            return 0.0

        if text[-1] in self.SI_PREFIXES:
            prefix = text[-1]
            number = text[:-1]
            if not number:
                raise ValueError("Invalid number format")
            try:
                self.current_prefix = prefix
                value = float(number) * self.SI_PREFIXES[prefix]
                return value
            except ValueError:
                raise ValueError("Invalid number format")

        try:
            self.current_prefix = None
            return float(text)
        except ValueError:
            raise ValueError("Invalid number format")

    def keyPressEvent(self, event):
        """Handle precise value adjustments based on cursor position."""
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            cursor_pos = self.lineEdit().cursorPosition()
            text = self.lineEdit().text()

            # Get the numeric part and any SI prefix
            if text and text[-1] in self.SI_PREFIXES:
                number = text[:-1]
                prefix = text[-1]
            else:
                number = text
                prefix = ""

            # Find the position relative to the decimal point
            if "." in number:
                decimal_pos = number.find(".")
                if cursor_pos > decimal_pos:
                    # After decimal point: count positions after decimal
                    power = -(cursor_pos - decimal_pos)
                else:
                    # Before decimal point: count from right to cursor
                    power = len(number[:decimal_pos]) - cursor_pos - 1
            else:
                # For integers: count from right to cursor
                power = len(number) - cursor_pos - 1

            # Calculate step size based on cursor position
            step = 10**power

            # Get the current value and preserve decimal places
            decimal_places = len(number.split(".")[-1]) if "." in number else 0
            current_value = float(number)

            # Apply the step to the unscaled value
            if event.key() == Qt.Key.Key_Up:
                new_value = current_value + step
            else:
                new_value = current_value - step

            # Format to preserve decimal places
            if decimal_places > 0:
                format_str = f"{{:.{decimal_places}f}}"
                new_value = float(format_str.format(new_value))

            # Preserve the prefix by applying its scaling
            if prefix:
                new_value = new_value * self.SI_PREFIXES[prefix]

            # Set the new value
            self.setValue(new_value)

            # Restore cursor position
            self.lineEdit().setCursorPosition(cursor_pos)
        else:
            super().keyPressEvent(event)

    def stepBy(self, steps: int):
        """Handle spinbox arrow buttons."""
        text = self.lineEdit().text()

        # Get the numeric part and any SI prefix
        if text and text[-1] in self.SI_PREFIXES:
            number = text[:-1]
            prefix = text[-1]
        else:
            number = text
            prefix = ""

        # Calculate step size based on rightmost digit
        if "." in number:
            decimal_places = len(number.split(".")[-1])
            power = -decimal_places
        else:
            power = 0

        step = 10**power

        # Get current value and apply step
        current_value = float(number)
        new_value = current_value + (step * steps)

        # Format to preserve decimal places
        if "." in number:
            format_str = f"{{:.{decimal_places}f}}"
            new_value = float(format_str.format(new_value))

        # Apply prefix scaling if needed
        if prefix:
            new_value = new_value * self.SI_PREFIXES[prefix]

        # Set the new value
        self.setValue(new_value)

    def textFromValue(self, value: float) -> str:
        """Convert value to text, maintaining SI prefix notation."""
        if value == 0:
            return "0"

        abs_value = abs(value)
        exponent = int(f"{abs_value:e}".split("e")[1])
        prefix_exp = 3 * (exponent // 3)

        if self.current_prefix is not None:
            user_scale = self.SI_PREFIXES[self.current_prefix]
            user_exp = int(f"{user_scale:e}".split("e")[1])
            if abs(user_exp - prefix_exp) <= 3:
                scaled_value = value / user_scale
                return f"{scaled_value:g}{self.current_prefix}"

        scale = 10**prefix_exp
        prefix = self.REVERSE_PREFIXES.get(scale, "")

        if prefix:
            scaled_value = value / scale
            return f"{scaled_value:g}{prefix}"

        return f"{value:g}"
