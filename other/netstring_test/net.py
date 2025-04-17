

import asyncio
import netstring

# 4096 byte chunks...
async def source(data):
    while data:
        evt, data = data[:4096], data[4096:]
        yield evt

async def test_async():
    src = source(b"4:good,5:foo,") # expect b"good" out
    
    strm = netstring.async_stream_payload(src)

#     async while True:
#         try:
#             event = next(strm)
#         except ValueError as e:
#             
    events = []
    async for event in strm:
        events.append(event)
    return events
    

res = asyncio.run(test_async())
print(res)

b = b'51:\x82\xa7command\xa5hello\xa6params\x82\xa4blaa\xa4blaa\xaaimage_size\x92\xcd\n\x00\xcd\n\x00,'
print(netstring.decode(b))
