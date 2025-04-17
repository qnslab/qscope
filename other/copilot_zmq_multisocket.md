Yes, you can monitor multiple ZeroMQ sockets using file descriptors (`zmq.FD`) for events. You will need to create a `QSocketNotifier` for each socket you want to monitor. Here is a step-by-step plan and the corresponding code:

1. **Create a list to hold the `QSocketNotifier` instances.**
2. **Initialize each ZeroMQ socket and bind/connect them as needed.**
3. **Create a `QSocketNotifier` for each socket and connect its `activated` signal to a slot that handles the events.**
4. **In the event handler slot, disable the notifier, process the message, and re-enable the notifier.**

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
    def __init__(self, topics, ports, mainwindow, parent=None):
        super(Server, self).__init__(parent)
        self._zmq_context = zmq.Context()
        self._zmq_socks = []
        self._zmq_notifiers = []

        for port in ports:
            zmq_sock = self._zmq_context.socket(zmq.SUB)
            zmq_sock.bind(f"tcp://*:{port}")
            for topic in topics:
                zmq_sock.setsockopt(zmq.SUBSCRIBE, topic)
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
        if sock_status == zmq.POLLIN:
            msg = zmq_sock.recv_multipart()
            topic = msg[0]
            callback = self._topic_map[topic]
            callback(msg)
        notifier.setEnabled(True)
        zmq_sock.getsockopt(zmq.EVENTS)

    def _onQuit(self):
        for notifier in self._zmq_notifiers:
            notifier.activated.disconnect(self._onZmqMsgRecv)
            notifier.setEnabled(False)
        self._zmq_context.destroy(0)
```

This code sets up multiple ZeroMQ sockets and monitors them using `QSocketNotifier` instances. Each notifier is connected to the `_onZmqMsgRecv` slot, which processes incoming messages.