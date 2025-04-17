import zmq
import time
import os

# Enable ZMQ logging
os.environ['ZMQ_LOG'] = '1'

def main():
    context = zmq.Context()

    # Set up the first PUB socket
    pub_socket1 = context.socket(zmq.PUB)
    pub_socket1.bind("tcp://localhost:5555")

    # Set up the second PUB socket
    pub_socket2 = context.socket(zmq.PUB)
    pub_socket2.bind("tcp://localhost:5556")

    # Set up the ROUTER socket
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind("tcp://localhost:5557")

    print("Server is running...")

    while True:
        # Send a multipart message on the first PUB socket
        topic1 = b"topic1"
        message1 = b"Hello from PUB socket 1"
        pub_socket1.send_multipart([topic1, message1])
        print("Sent multipart message on PUB socket 1")

        # Send a multipart message on the second PUB socket
        topic2 = b"topic2"
        message2 = b"Hello from PUB socket 2"
        pub_socket2.send_multipart([topic2, message2])
        print("Sent multipart message on PUB socket 2")

        # Check for requests on the ROUTER socket
        try:
            # Receive the entire message (identity, empty frame, and message)
            identity, empty, message = router_socket.recv_multipart(flags=zmq.NOBLOCK)
            print(f"Received request on ROUTER socket: {message}")
            response = f"Response to: {message}"
            # Send the response back to the client
            router_socket.send_multipart([identity, b'', response.encode()])
        except zmq.Again:
            print("No message received on ROUTER socket")
            pass

        time.sleep(1)

if __name__ == "__main__":
    main()