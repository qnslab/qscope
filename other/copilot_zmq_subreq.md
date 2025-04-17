### Plan

1. **Create a list to hold the `QSocketNotifier` instances.**
2. **Initialize each ZeroMQ socket (SUB and REQ) and bind/connect them as needed.**
3. **Create a `QSocketNotifier` for each socket and connect its `activated` signal to a slot that handles the events.**
4. **In the event handler slot, disable the notizfier, process the message, and re-enable the notifier.**
5. **Add logic to handle different types of events for each ZeroMQ socket.**

### Code

```python
import zmq
from PyQt4 import QtCore, QtGui

class QZmqSocketNotifier(QtCore.QSocketNotifier):
    """ Provides Qt event notifier for ZMQ socket events """
    def __init__(self, zmq_sock, event_type, parent=None):
        super(QZmqSocketNotifier, self).__init__(zmq_sock.getsockopt(zmq.FD), event_type, parent)
        self.zmq_sock = zmq_sock

class Server(QtGui.QFrame):
    def __init__(self, sub_topics, sub_ports, req_ports, mainwindow, parent=None):
        super(Server, self).__init__(parent)
        self._zmq_context = zmq.Context()
        self._zmq_socks = []
        self._zmq_notifiers = []

        # Initialize SUB sockets
        for port in sub_ports:
            zmq_sock = self._zmq_context.socket(zmq.SUB)
            zmq_sock.bind(f"tcp://*:{port}")
            for topic in sub_topics:
                zmq_sock.setsockopt(zmq.SUBSCRIBE, topic)
            self._zmq_socks.append(zmq_sock)
            zmq_notifier = QZmqSocketNotifier(zmq_sock, QtCore.QSocketNotifier.Read)
            zmq_notifier.activated.connect(self._onZmqMsgRecv)
            self._zmq_notifiers.append(zmq_notifier)

        # Initialize REQ sockets
        for port in req_ports:
            zmq_sock = self._zmq_context.socket(zmq.REQ)
            zmq_sock.connect(f"tcp://localhost:{port}")
            self._zmq_socks.append(zmq_sock)
            zmq_notifier = QZmqSocketNotifier(zmq_sock, QtCore.QSocketNotifier.Read)
            zmq_notifier.activated.connect(self._onZmqMsgRecv)
            self._zmq_notifiers.append(zmq_notifier)

        mainwindow.quit.connect(self._onQuit)

    @QtCore.pyqtSlot()
    def _onZmqMsgRecv(self):
        notifier = self.sender()
        notifier.setEnabled(False)
        zmq_sock = notifier.zmq_sock
        sock_status = zmq_sock.getsockopt(zmq.EVENTS)
        if sock_status & zmq.POLLIN:
            self._handle_recv_event(zmq_sock)
        if sock_status & zmq.POLLOUT:
            self._handle_send_event(zmq_sock)
        if sock_status & zmq.POLLERR:
            self._handle_error_event(zmq_sock)
        notifier.setEnabled(True)

    def _handle_recv_event(self, zmq_sock):
        if zmq_sock.getsockopt(zmq.TYPE) == zmq.SUB:
            msg = zmq_sock.recv_multipart()
            topic = msg[0]
            callback = self._topic_map[topic]
            callback(msg)
        elif zmq_sock.getsockopt(zmq.TYPE) == zmq.REQ:
            msg = zmq_sock.recv()
            # Handle REQ socket message
            print(f"Received REQ message: {msg}")

    def _handle_send_event(self, zmq_sock):
        # Handle send event
        pass

    def _handle_error_event(self, zmq_sock):
        # Handle error event
        pass

    def _onQuit(self):
        for notifier in self._zmq_notifiers:
            notifier.activated.disconnect(self._onZmqMsgRecv)
            notifier.setEnabled(False)
        self._zmq_context.destroy(0)
```