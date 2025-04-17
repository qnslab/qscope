import numpy as np
import matplotlib.pyplot as plt
import simplejpeg
import asyncio
import zmq
import zmq.asyncio

# print(f"Current libzmq version is {zmq.zmq_version()}")
# print(f"Current  pyzmq version is {zmq.__version__}")

async def plotter(plotting_queue, stop_event):
    plt.show(block=False    )
    fig, ax = plt.subplots()

    while True:
        await asyncio.sleep(0) # Small pause to prevent busy waiting.
        if stop_event.is_set():
            plt.close(fig)
            break
        if not plotting_queue.empty():
            try:
                image = plotting_queue.get_nowait()
                ax.clear()
                ax.imshow(image, cmap="gray")
                ax.set_xticks([])
                ax.set_yticks([])
                plt.draw()
                plt.pause(0.1)
            except asyncio.QueueEmpty:
                continue


async def main(qu, stop_event):
    ctx = zmq.asyncio.Context()
    # ctx = zmq.Context()
    msg_channel = ctx.socket(zmq.REQ)
    msg_channel.connect("tcp://127.0.0.1:5555")
    video_channel = ctx.socket(zmq.SUB)
    video_channel.connect("tcp://127.0.0.1:5556")
    video_channel.setsockopt(zmq.SUBSCRIBE, b"") # subscribe to all messages
    video_channel.subscribe(b"")

    # Initialize poll set
    poller = zmq.Poller()
    poller.register(msg_channel, zmq.POLLIN)
    poller.register(video_channel, zmq.POLLIN)

    sending_lst = [b"blaa", b"blaa2", b"blaa3", b"start", b"foo", b"bar", b"oop",
                   b"goop", b"stop"]
    awaiting_resp = False
    while True:
        if sending_lst and not awaiting_resp: # don't send if we're waiting for a response
            msg = sending_lst.pop(0)
            await msg_channel.send(msg)
            awaiting_resp = True
        try:
            socks = dict(poller.poll()) # may want to timeout here
        except (KeyboardInterrupt,   SystemExit):
            break
        # print(socks)
        if msg_channel in socks:
            resp = await msg_channel.recv()
            print(f"Msg: {msg}, Resp: {resp}")
            assert resp == msg
            awaiting_resp = False
            if resp == b"stop":
                stop_event.set()
                break
        if video_channel in socks:
            jpg_buffer = await video_channel.recv(flags=0, copy=False, track=False)
            frame = simplejpeg.decode_jpeg(jpg_buffer, colorspace='GRAY')
            qu.put_nowait(frame)
        await asyncio.sleep(0.1)


async def wrapper(plotting_queue):
    stop_event = asyncio.Event()
    cor1 = plotter(plotting_queue, stop_event)
    cor2 = main(plotting_queue, stop_event)
    await asyncio.gather(cor1, cor2)

if __name__ == '__main__':
    plotting_queue = asyncio.Queue()
    asyncio.run(wrapper(plotting_queue))
