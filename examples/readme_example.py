import matplotlib.pyplot as plt
import qscope.server
import qscope.system
import qscope.types
from qscope.scripting import meas_close_after_nsweeps

NUM_SWEEPS = 2

qscope.server.start_client_log() # log client messages to ~./qscope/client.log

manager = qscope.server.ConnectionManager()
manager.start_local_server(  # starts a (local) server in a new process
    "mock",
) # logs go to ~./qscope/server.log

manager.connect()
manager.startup() # boot hardware

config = qscope.types.TESTING_MEAS_CONFIG # configuration for a measurement
meas_id = manager.add_measurement(config)
manager.start_measurement_wait(meas_id)

sweep_data = meas_close_after_nsweeps(manager, meas_id, NUM_SWEEPS)

manager.stop_local_server()

fig, ax = plt.subplots()
x, y_sig, y_ref = sweep_data
ax.plot(x, y_sig, "-o", label="Signal")
ax.plot(x, y_ref, "-o", label="Reference")
ax.legend()
plt.show()
