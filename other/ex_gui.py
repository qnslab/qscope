# make a simple PyQt6 UI that generates a lorentzian with noise as a slow and long task
# the long task will be performed in a separate thread to allow for the GUI plotting
# to update asynchronously with the current state of the data in the long task

import sys, traceback, time
import numpy as np
import matplotlib
matplotlib.use('QtAgg')

import asyncio

from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matplotlib + PyQt6")
        self.resize(800, 600)

        # create a layout
        layout = QVBoxLayout()
        self.color = 'b'

        self.canvas = FigureCanvasQTAgg(Figure())

        self.ax = self.canvas.figure.subplots()
        self.ax.set_xlim(0, 100)

        self.x_data = np.linspace(0, 100, 100)
        self.y_data = np.zeros(100)

        self.timer = QTimer()
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

        self.start_button = QPushButton("Start Measurement")
        self.start_button.clicked.connect(self.start_measurement)

        # add a button to change the color of the line
        self.button = QPushButton("Change Color")
        self.button.clicked.connect(self.change_color)

        # add an indicator to count the number of updates
        self.progress = QProgressBar()
        self.progress.setValue(0)

        layout.addWidget(self.start_button)
        layout.addWidget(self.canvas)
        layout.addWidget(self.button)
        layout.addWidget(self.progress)

        # create a widget to hold the layout
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)


        self.threadpool = QThreadPool()
        print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())


    def start_measurement(self):
        # start the long task in a separate thread tgar periodically gets the data
        # Pass the function to execute
        worker = Worker(self.get_data) # Any other args, kwargs are passed to the run function
        # worker.signals.result.connect(self.get_data)
        worker.signals.finished.connect(self.thread_complete)
        worker.signals.progress.connect(self.progress_fn)

        # Execute
        self.threadpool.start(worker)
        return

    def thread_complete(self):
        print("MEASUREMENT COMPLETE!")

    def progress_fn(self, n):
        self.progress.setValue(n)

    async def get_data(self, *args, **kwargs):
        # get the data from the long task
        n_loops = 100
        for i in range(n_loops):
            self.y_data = 1/(1 + (self.x_data - 50)**2) + 0.1*np.random.randn(100)
            time.sleep(.01)
            # update the graph
            self.update_plot()

            # use the progress callback to update the progress bar
            kwargs['progress_callback'].emit(int(100*i/(n_loops - 1)))

            # update the progress bar
            # self.progress.setValue(int(100*i/n_loops))
        # end the worker thread
        return self.y_data


    def update_plot(self):
        # update the plot with the current state of the data
        self.ax.clear()
        self.ax.plot(self.x_data, self.y_data, color=self.color)
        self.canvas.draw()

    def change_color(self):
        # change the color of the line
        self.color = 'b' if self.color != 'b' else 'r'
        # if self.color.isValid():
        self.ax.lines[0].set_color(self.color)
        self.canvas.draw()



class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = asyncio.run(self.fn(*self.args, **self.kwargs))
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(int)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()

