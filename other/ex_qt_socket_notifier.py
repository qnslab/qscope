# https://stackoverflow.com/a/32051472/11314260
import zmq
from PyQt4 import QtCore, QtGui

class QZmqSocketNotifier( QtCore.QSocketNotifier ):
    """ Provides Qt event notifier for ZMQ socket events """
    def __init__( self, zmq_sock, event_type, parent=None ):
        """
        Parameters:
        ----------
        zmq_sock : zmq.Socket
            The ZMQ socket to listen on. Must already be connected or bound to a socket
            address.
        event_type : QtSocketNotifier.Type
            Event type to listen for, as described in documentation for QtSocketNotifier
        """
        super( QZmqSocketNotifier, self ).__init__( zmq_sock.getsockopt(zmq.FD),
                                                    event_type, parent )

class Server(QtGui.QFrame):

    def __init__(self, topics, port, mainwindow, parent=None):
        super(Server, self).__init__(parent)

        self._PORT = port

        # Create notifier to handle ZMQ socket events coming from client
        self._zmq_context = zmq.Context()
        self._zmq_sock = self._zmq_context.socket( zmq.SUB )
        self._zmq_sock.bind( "tcp://*:" + self._PORT )
        for topic in topics:
            self._zmq_sock.setsockopt( zmq.SUBSCRIBE, topic )
        self._zmq_notifier = QZmqSocketNotifier( self._zmq_sock,
                                                 QtCore.QSocketNotifier.Read )

        # connect signals and slots
        self._zmq_notifier.activated.connect( self._onZmqMsgRecv )
        mainwindow.quit.connect( self._onQuit )

    @QtCore.pyqtSlot()
    def _onZmqMsgRecv(self):
        self._zmq_notifier.setEnabled(False)
        # Verify that there's data in the stream
        sock_status = self._zmq_sock.getsockopt( zmq.EVENTS )
        if sock_status == zmq.POLLIN:
            msg = self._zmq_sock.recv_multipart()
            topic = msg[0]
            callback = self._topic_map[ topic ]
            callback( msg )
        self._zmq_notifier.setEnabled(True)
        self._zmq_sock.getsockopt(zmq.EVENTS)

    def _onQuit(self):
        self._zmq_notifier.activated.disconnect( self._onZmqMsgRecv )
        self._zmq_notifier.setEnabled(False)
        del self._zmq_notifier
        self._zmq_context.destroy(0)

# Disabling and then re-enabling the notifier in _on_ZmqMsgRecv is per the documentation
# for QSocketNotifier.
# The final call to getsockopt is for some reason necessary. Otherwise, the notifier
# stops working after the first event. I was actually going to post a new question for
# this. Does anyone know why this is needed?
#  [SAM] this is the same in https://gist.github.com/ivanalejandro0/dba1dd504edbc2c32e83
# Note that if you don't destroy the notifier before the ZMQ context, you'll probably
# get an error like this when you quit the application:
# QSocketNotifier: Invalid socket 16 and type 'Read', disabling...
