import matplotlib.pyplot as plt
import qscope.server
import qscope.system
import qscope.types
from qscope.scripting import meas_close_after_nsweeps

NUM_SWEEPS = 2

qscope.server.start_client_log() # log client messages to ~./qscope/client.log

manager = qscope.server.ConnectionManager()
manager.start_local_server(  # starts a (local) server in a new process
    # "mock",
    "gmx",
) # logs go to ~./qscope/server.log

manager.connect()
manager.startup() # boot hardware

manager.camera_set_params(30e-3, (2048, 2048), (1,1))
snapshot = manager.camera_take_snapshot()

manager.stop_local_server()

fig, ax = plt.subplots()
ax.imshow(snapshot, cmap="gray_r")
plt.show()
