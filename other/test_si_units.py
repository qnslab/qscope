import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QLabel, QDoubleSpinBox)
from PyQt6.QtGui import QValidator
from PyQt6.QtCore import Qt


from typing import Tuple, Optional
from PyQt6.QtGui import QValidator

class SIPrefixSpinBox(QDoubleSpinBox):
    """SpinBox that accepts SI prefixes in input.
    
    A custom QDoubleSpinBox that handles SI prefix notation (like k, M, G etc.)
    for number input and display. Supports keyboard interaction for precise 
    value adjustments.
    
    Attributes:
        SI_PREFIXES: Dict mapping prefix chars to their scale factors
        REVERSE_PREFIXES: Dict mapping scale factors to prefix chars
        current_prefix: Currently active SI prefix, if any
    """

    SI_PREFIXES = {
        'f': 1e-15,
        'p': 1e-12,
        'n': 1e-9,
        'u': 1e-6,
        'm': 1e-3,
        'k': 1e3,
        'M': 1e6,
        'G': 1e9,
        'T': 1e12
    }

    REVERSE_PREFIXES = {v: k for k, v in SI_PREFIXES.items()}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)
        self.setDecimals(10)  # Allow more decimal places
        self.step_scale = 1.0
        self.current_prefix: Optional[str] = None  # Track user's chosen prefix
        self.setSingleStep(0.0)  # We'll handle steps manually
        
        # Add tooltip showing actual value
        self.valueChanged.connect(self._update_tooltip)
        
    def _update_tooltip(self, value: float) -> None:
        """Update tooltip to show the actual value."""
        self.setToolTip(f"Actual value: {value:g}")
        
    def get_prefix(self) -> Optional[str]:
        """Get the current SI prefix."""
        return self.current_prefix
        
    def set_prefix(self, prefix: Optional[str]) -> None:
        """Set the SI prefix to use for display.
        
        Args:
            prefix: SI prefix character or None to auto-select
        Raises:
            ValueError: If prefix is invalid
        """
        if prefix is not None and prefix not in self.SI_PREFIXES:
            raise ValueError(f"Invalid SI prefix: {prefix}")
        self.current_prefix = prefix
        self.setValue(self.value())  # Trigger display update

    def keyPressEvent(self, event):
        # only handle arrow keys up/down
        if event.key() in (Qt.Key.Key_Up, Qt.Key.Key_Down):
            cursor_pos = self.lineEdit().cursorPosition()
            text = self.lineEdit().text()
            
            # Get the numeric part and any SI prefix
            if text and text[-1] in self.SI_PREFIXES:
                number = text[:-1]
                prefix = text[-1]
            else:
                number = text
                prefix = ''
            
            # Find the position relative to the decimal point
            if '.' in number:
                decimal_pos = number.find('.')
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
            step = 10 ** power
            
            # Get the current value and preserve decimal places
            decimal_places = len(number.split('.')[-1]) if '.' in number else 0
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

    def validate(self, text: str, pos: int) -> Tuple[QValidator.State, str, int]:
        """Validate input text.
        
        Args:
            text: Input text to validate
            pos: Cursor position
            
        Returns:
            Tuple of (validation state, text, cursor position)
        """
        # Allow negative numbers
        valid_chars = set("0123456789.-" + "".join(self.SI_PREFIXES.keys()))
        
        # Only allow one decimal point and one minus sign at start
        if text.count('.') > 1 or (text.count('-') > 0 and text[0] != '-'):
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

        # Handle SI prefix
        if text[-1] in self.SI_PREFIXES:
            prefix = text[-1]
            number = text[:-1]
            if not number:  # Handle case of just a prefix
                raise ValueError("Invalid number format")
            try:
                self.current_prefix = prefix  # Store the user's chosen prefix
                value = float(number) * self.SI_PREFIXES[prefix]
                return value
            except ValueError:
                raise ValueError("Invalid number format")
        
        # No prefix case
        try:
            self.current_prefix = None  # Clear prefix if none used
            return float(text)
        except ValueError:
            raise ValueError("Invalid number format")

    def stepBy(self, steps: int):
        """Handle spinbox arrow buttons."""
        text = self.lineEdit().text()
        
        # Get the numeric part and any SI prefix
        if text and text[-1] in self.SI_PREFIXES:
            number = text[:-1]
            prefix = text[-1]
        else:
            number = text
            prefix = ''
        
        # Calculate step size based on rightmost digit
        if '.' in number:
            decimal_places = len(number.split('.')[-1])
            power = -decimal_places
        else:
            power = 0
            
        step = 10 ** power
        
        # Get current value and apply step
        current_value = float(number)
        new_value = current_value + (step * steps)
        
        # Format to preserve decimal places
        if '.' in number:
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
        exponent = int(f"{abs_value:e}".split('e')[1])
        # Round to nearest multiple of 3 for the exponent
        prefix_exp = 3 * (exponent // 3)
        
        # If user has chosen a prefix, check if it's within one order of the ideal prefix
        if self.current_prefix is not None:
            user_scale = self.SI_PREFIXES[self.current_prefix]
            user_exp = int(f"{user_scale:e}".split('e')[1])
            # Only use user prefix if it's within ±1 order of magnitude of ideal
            if abs(user_exp - prefix_exp) <= 3:
                scaled_value = value / user_scale
                return f"{scaled_value:g}{self.current_prefix}"

        # Find the appropriate prefix based on the rounded exponent
        scale = 10 ** prefix_exp
        prefix = self.REVERSE_PREFIXES.get(scale, '')
        
        if prefix:
            scaled_value = value / scale
            return f"{scaled_value:g}{prefix}"
        
        return f"{value:g}"


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SI Prefix SpinBox Demo")

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add instruction label
        instruction_label = QLabel("Enter a value with SI prefix (e.g., 10u, 1.5M, 2n):")
        layout.addWidget(instruction_label)

        # Create the SI prefix spinbox
        self.spinbox = SIPrefixSpinBox()
        self.spinbox.setRange(-1e15,
                              1e15)  # Set a wide range
        self.spinbox.valueChanged.connect(self.update_value_label)
        layout.addWidget(self.spinbox)

        # Create label to show the current value
        self.value_label = QLabel("Current value: 0")
        layout.addWidget(self.value_label)

        # Set some reasonable window defaults
        self.setMinimumWidth(300)

    def update_value_label(self, value):
        # Display both the raw value and the SI notation
        si_text = self.spinbox.textFromValue(value)
        self.value_label.setText(f"Current value: {si_text} ({value:g})")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
