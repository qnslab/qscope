import os
import sys

from PyQt6 import QtCore
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from qscope.gui.main_window import MainWindow
from qscope.util import start_client_log


def main_gui(
    system_name=None,
    host="",
    msg_port="",
    log_to_file=True,
    log_to_stdout=True,
    log_path=None,
    clear_prev_log=True,
    log_level="INFO",
    auto_connect=True,
):
    start_client_log(
        log_to_file=log_to_file,
        log_to_stdout=log_to_stdout,
        log_path=log_path,
        clear_prev=clear_prev_log,
        log_level=log_level,
    )
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow(
        system_name=system_name, auto_connect=auto_connect, host=host, msg_port=msg_port
    )
    window.show()

    # Return the exit code instead of calling sys.exit directly
    return_code = app.exec()

    # Ensure all threads are stopped and resources are cleaned up
    app.quit()  # Make sure the application knows it should quit

    # Force Python garbage collection
    import gc

    gc.collect()

    # If there are any remaining ZMQ contexts, terminate them
    try:
        import zmq

        zmq.Context.instance().term()
    except:
        pass

    # If there are any remaining asyncio event loops, close them
    try:
        import asyncio

        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.stop()
        loop.close()
    except:
        pass

    if __name__ != "__main__":  # Don't exit if being imported (e.g. during testing)
        return return_code
    sys.exit(return_code)


if __name__ == "__main__":
    main_gui()
