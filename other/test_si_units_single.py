import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget,
                             QVBoxLayout, QLabel, QSpinBox)
from PyQt6.QtGui import QValidator
from PyQt6.QtCore import Qt
from typing import Tuple, Optional

class SIPrefixSpinBox(QSpinBox):
    """SpinBox that accepts SI prefixes in input.
    
    A custom QSpinBox that handles SI prefix notation (like k, M, G etc.)
    for integer input and display. Supports keyboard interaction for precise 
    value adjustments.
    
    Attributes:
        SI_PREFIXES: Dict mapping prefix chars to their scale factors
        REVERSE_PREFIXES: Dict mapping scale factors to prefix chars
        current_prefix: Currently active SI prefix, if any
    """

    SI_PREFIXES = {
        'k': 1000,
        'M': 1000000,
        'G': 1000000000,
        'T': 1000000000000
    }

    REVERSE_PREFIXES = {v: k for k, v in SI_PREFIXES.items()}

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setKeyboardTracking(False)
        self.current_prefix: Optional[str] = None  # Track user's chosen prefix
        self.setSingleStep(0)  # We'll handle steps manually
        
        # Add tooltip showing actual value
        self.valueChanged.connect(self._update_tooltip)
        
    def _update_tooltip(self, value: int) -> None:
        """Update tooltip to show the actual value."""
        self.setToolTip(f"Actual value: {value:d}")
        
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
            
            # Calculate step size based on cursor position
            power = len(number) - cursor_pos - 1
            if power < 0:
                power = 0
            step = 10 ** power
            
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

    def validate(self, text: str, pos: int) -> Tuple[QValidator.State, str, int]:
        """Validate input text.
        
        Args:
            text: Input text to validate
            pos: Cursor position
            
        Returns:
            Tuple of (validation state, text, cursor position)
        """
        # Allow negative numbers
        valid_chars = set("0123456789-" + "".join(self.SI_PREFIXES.keys()))
        
        # Only allow one minus sign at start
        if text.count('-') > 0 and text[0] != '-':
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

        # Handle SI prefix
        if text[-1] in self.SI_PREFIXES:
            prefix = text[-1]
            number = text[:-1]
            if not number:  # Handle case of just a prefix
                raise ValueError("Invalid number format")
            try:
                self.current_prefix = prefix  # Store the user's chosen prefix
                value = int(float(number) * self.SI_PREFIXES[prefix])
                return value
            except ValueError:
                raise ValueError("Invalid number format")
        
        # No prefix case
        try:
            self.current_prefix = None  # Clear prefix if none used
            return int(text)
        except ValueError:
            raise ValueError("Invalid number format")

    def stepBy(self, steps: int) -> None:
        """Handle spinbox arrow buttons."""
        text = self.lineEdit().text()
        
        # Get the numeric part and any SI prefix
        if text and text[-1] in self.SI_PREFIXES:
            number = text[:-1]
            prefix = text[-1]
        else:
            number = text
            prefix = ''
        
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
        
        # Find the appropriate prefix
        for scale, prefix in sorted(self.REVERSE_PREFIXES.items(), reverse=True):
            if abs_value >= scale and abs_value % (scale // 10) == 0:
                scaled_value = value // scale
                return f"{scaled_value}{prefix}"
        
        return str(value)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SI Prefix SpinBox Demo (Integer)")

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Add instruction label
        instruction_label = QLabel("Enter an integer value with SI prefix (e.g., 10k, 2M, 3G):")
        layout.addWidget(instruction_label)

        # Create the SI prefix spinbox
        self.spinbox = SIPrefixSpinBox()
        self.spinbox.setRange(-2147483648, 2147483647)  # Set max 32-bit integer range
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
        self.value_label.setText(f"Current value: {si_text} ({value:d})")


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
