import numpy as np
import matplotlib.pyplot as plt
import simplejpeg
import threading
# import asyncio
import zmq
# import zmq.asyncio

import queue
import time

# print(f"Current libzmq version is {zmq.zmq_version()}")
# print(f"Current  pyzmq version is {zmq.__version__}")

def plotter(plotting_queue):
    plt.ion()
    fig, ax = plt.subplots()

    while True:
        if not plotting_queue.empty():
            image = plotting_queue.get()
            ax.clear()
            ax.imshow(image)
            plt.draw()
            plt.pause(0.1)
        plt.pause(0.1)  # Small pause to prevent busy waiting.


def main(qu):
    # ctx = zmq.asyncio.Context()
    ctx = zmq.Context()
    requester = ctx.socket(zmq.REQ)
    requester.connect("tcp://127.0.0.1:5555")
    video_subscriber = ctx.socket(zmq.SUB)
    video_subscriber.connect("tcp://127.0.0.1:5556")
    video_subscriber.setsockopt(zmq.SUBSCRIBE, b"") # subscribe to all messages
    video_subscriber.subscribe(b"")

    # Initialize poll set
    poller = zmq.Poller()
    poller.register(requester, zmq.POLLIN)
    poller.register(video_subscriber, zmq.POLLIN)

    sending_lst = [b"blaa", b"blaa2", b"blaa3", b"start", b"foo", b"bar", b"oop",
                   b"goop", b"stop"]
    awaiting_resp = False
    while True:
        if sending_lst and not awaiting_resp: # don't send if we're waiting for a response
            msg = sending_lst.pop(0)
            requester.send(msg)
            print(f"Sent {msg}")
            awaiting_resp = True
        try:
            socks = dict(poller.poll()) # may want to timeout here
        except (KeyboardInterrupt, SystemExit):
            break
        # print(socks)
        if requester in socks:
            resp = requester.recv()
            print(msg, resp)
            awaiting_resp = False
        if video_subscriber in socks:
            jpg_buffer = video_subscriber.recv(flags=0, copy=False, track=False)
            frame = simplejpeg.decode_jpeg(jpg_buffer, colorspace='GRAY')
            qu.put(frame)
        time.sleep(0.1)


if __name__ == '__main__':
    plotting_queue = queue.Queue()
    t = threading.Thread(target=main, args=(plotting_queue  ,))
    t.start()
    plotter(plotting_queue)
    t.join()