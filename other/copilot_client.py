import zmq
from PyQt5 import QtCore, QtWidgets

class QZmqSocketNotifier(QtCore.QSocketNotifier):
    """ Provides Qt event notifier for ZMQ socket events """
    def __init__(self, zmq_sock, event_type, parent=None):
        super(QZmqSocketNotifier, self).__init__(zmq_sock.getsockopt(zmq.FD), event_type, parent)
        self.zmq_sock = zmq_sock

class ServerComms(QtWidgets.QFrame):
    def __init__(self, context, sub_topics, sub_ports, req_port, parent=None):
        super(ServerComms, self).__init__(parent)
        self._zmq_context = context
        self._zmq_socks = []
        self._zmq_notifiers = []

        # Initialize SUB sockets
        for port in sub_ports:
            zmq_sock = self._zmq_context.socket(zmq.SUB)
            zmq_sock.connect(f"tcp://localhost:{port}")
            for topic in sub_topics:
                zmq_sock.setsockopt(zmq.SUBSCRIBE, topic)
            self._zmq_socks.append(zmq_sock)
            zmq_notifier = QZmqSocketNotifier(zmq_sock, QtCore.QSocketNotifier.Read)
            zmq_notifier.activated.connect(self._onZmqMsgRecv)
            self._zmq_notifiers.append(zmq_notifier)

        # Initialize a single REQ socket
        self._req_sock = self._zmq_context.socket(zmq.REQ)
        self._req_sock.connect(f"tcp://localhost:{req_port}")

        # GUI elements
        self.input_textbox = QtWidgets.QLineEdit(self)
        self.input_textbox.installEventFilter(self)  # Add event filter
        self.send_button = QtWidgets.QPushButton("Send", self)
        self.send_button.clicked.connect(self.send_req_message)
        self.output_textbox = QtWidgets.QTextEdit(self)
        self.output_textbox.setReadOnly(True)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.input_textbox)
        layout.addWidget(self.send_button)
        layout.addWidget(self.output_textbox)
        self.setLayout(layout)

    def eventFilter(self, source, event):
        if event.type() == QtCore.QEvent.KeyPress and source is self.input_textbox:
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self.send_button.click()
                return True
        return super(ServerComms, self).eventFilter(source, event)

    @QtCore.pyqtSlot()
    def _onZmqMsgRecv(self):
        notifier = self.sender()
        notifier.setEnabled(False)
        zmq_sock = notifier.zmq_sock
        sock_status = zmq_sock.getsockopt(zmq.EVENTS)
        if sock_status & zmq.POLLIN:
            self._handle_recv_event(zmq_sock)
        notifier.setEnabled(True)
        zmq_sock.getsockopt(zmq.EVENTS) # needed to allow more messages... not sure why

    def _handle_recv_event(self, zmq_sock):
        print("recieving!!")
        if zmq_sock.getsockopt(zmq.TYPE) == zmq.SUB:
            msg = zmq_sock.recv_multipart()
            topic = msg[0]
            message = msg[1]
            self.output_textbox.append(f"Topic: {topic}, Message: {message}")

    def send_req_message(self):
        message = self.input_textbox.text()
        print("REQ msg: %s" % message)
        self._req_sock.send_string(message)
        reply = self._req_sock.recv_string()
        self.output_textbox.append(f"ROUTER reply: {reply}")

    def _onQuit(self):
        for notifier in self._zmq_notifiers:
            notifier.activated.disconnect(self._onZmqMsgRecv)
            notifier.setEnabled(False)
        self._zmq_context.destroy(0)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, context, sub_topics, sub_ports, req_port):
        super(MainWindow, self).__init__()
        self.server_comms = ServerComms(context, sub_topics, sub_ports, req_port, self)
        self.setCentralWidget(self.server_comms)
        self.setWindowTitle("ZeroMQ SUB/REQ Example")
        self.resize(400, 300)

    def closeEvent(self, event):
        self.server_comms._onQuit()
        event.accept()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    context = zmq.Context()
    sub_topics = [b"topic1", b"topic2"]
    sub_ports = [5555, 5556]
    req_port = 5557
    main_window = MainWindow(context, sub_topics, sub_ports, req_port)
    main_window.show()
    sys.exit(app.exec_())