import numpy as np
import numpy.random
import simplejpeg

import time
# import asyncio
import zmq
# import zmq.asyncio

# print(f"Current libzmq version is {zmq.zmq_version()}")
# print(f"Current  pyzmq version is {zmq.__version__}")

def main():
    rng = np.random.default_rng()
    # ctx = zmq.asyncio.Context()
    ctx = zmq.Context()
    responder = ctx.socket(zmq.REP)
    responder.bind("tcp://127.0.0.1:5555")
    video_publisher = ctx.socket(zmq.PUB)
    video_publisher.bind("tcp://127.0.0.1:5556")

    # Initialize poll set
    poller = zmq.Poller()
    poller.register(responder, zmq.POLLIN)
    poller.register(video_publisher, zmq.POLLIN)

    send_video = False
    while True:
        try:
            socks = dict(poller.poll()) # may want to add timeout here.
        except (KeyboardInterrupt, SystemExit):
            break
        if responder in socks:
            resp = responder.recv()
            print(f"Recieved {resp}")
            responder.send(resp) # echo back
            if resp == b"start":
                send_video = True
            elif resp == b"stop":
                break
        if send_video:
            img = rng.integers(low=0, high=255, size=(256, 256, 1), dtype=np.uint8)
            jpg_buffer = simplejpeg.encode_jpeg(
                img,
                # quality=jpeg_quality,
                quality=0.85,
                colorspace="GRAY",
                colorsubsampling="Gray"
            )
            video_publisher.send(jpg_buffer, copy=False, track=False)
        time.sleep(0.1)

if __name__ == "__main__":
    main()