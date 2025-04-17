import numpy as np
import numpy.random
import simplejpeg

import time
import asyncio
import zmq
import zmq.asyncio

# print(f"Current libzmq version is {zmq.zmq_version()}")
# print(f"Current  pyzmq version is {zmq.__version__}")

async def main():
    rng = np.random.default_rng()
    ctx = zmq.asyncio.Context()
    # ctx = zmq.Context()
    msg_channel = ctx.socket(zmq.REP)
    msg_channel.bind("tcp://127.0.0.1:5555")
    video_channel = ctx.socket(zmq.PUB)
    video_channel.bind("tcp://127.0.0.1:5556")

    # Initialize poll set
    poller = zmq.Poller()
    poller.register(msg_channel, zmq.POLLIN)
    poller.register(video_channel, zmq.POLLIN)

    send_video = False
    while True:
        try:
            socks = dict(poller.poll()) # may want to add timeout here.
        except (KeyboardInterrupt, SystemExit):
            break
        if msg_channel in socks:
            resp = await msg_channel.recv()
            print(f"Recieved {resp}")
            await msg_channel.send(resp) # echo back
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
            video_channel.send(jpg_buffer, copy=False, track=False)
        await asyncio.sleep(0.1)

if __name__ == "__main__":
    asyncio.run(main())

# no real need to be async here...