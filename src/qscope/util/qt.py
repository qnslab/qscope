"""Qt-related utility functions."""


def process_qt_events():
    """Process Qt events if running in a Qt application.

    This function will process any pending Qt events if running within a Qt application context.
    If not in a Qt context (no Qt application instance or Qt not available), this is a no-op.
    """
    try:
        from PyQt6.QtWidgets import QApplication

        app = QApplication.instance()
        if app:
            app.processEvents()
    except ImportError:
        pass  # Not in Qt environment
