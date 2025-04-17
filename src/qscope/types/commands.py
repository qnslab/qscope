"""Command constants for client-server communication."""

import types

CONSTS = types.SimpleNamespace()

# Comms commands
CONSTS.COMMS = types.SimpleNamespace()
CONSTS.COMMS.SHUTDOWN = "CONSTS.COMMS.SHUTDOWN"
CONSTS.COMMS.PING = "CONSTS.COMMS.PING"
CONSTS.COMMS.PONG = "CONSTS.COMMS.PONG"
CONSTS.COMMS.ECHO = "CONSTS.COMMS.ECHO"
CONSTS.COMMS.GET_SERVER_LOG_PATH = "CONSTS.COMMS.GET_SERVER_LOG_PATH"
CONSTS.COMMS.STARTUP = "CONSTS.COMMS.STARTUP"
CONSTS.COMMS.PACKDOWN = "CONSTS.COMMS.PACKDOWN"
CONSTS.COMMS.GET_ALL_MEAS_INFO = "CONSTS.COMMS.GET_ALL_MEAS_INFO"
CONSTS.COMMS.CLIENT_SYNC = "CONSTS.COMMS.CLIENT_SYNC"
CONSTS.COMMS.GET_OTHER_PORTS = "CONSTS.COMMS.GET_OTHER_PORTS"
CONSTS.COMMS.IS_STREAMING = "CONSTS.COMMS.IS_STREAMING"
CONSTS.COMMS.GET_DEVICE_LOCKS = "CONSTS.COMMS.GET_DEVICE_LOCKS"
CONSTS.COMMS.SAVE_LATEST_STREAM = "CONSTS.COMMS.SAVE_LATEST_STREAM"
CONSTS.COMMS.SAVE_NOTES = "CONSTS.COMMS.SAVE_NOTES"

# Measurement commands
CONSTS.MEAS = types.SimpleNamespace()
CONSTS.MEAS.GET_STATE = "CONSTS.MEAS.GET_STATE"
CONSTS.MEAS.GET_INFO = "CONSTS.MEAS.GET_INFO"
CONSTS.MEAS.GET_SWEEP = "CONSTS.MEAS.GET_SWEEP"
CONSTS.MEAS.GET_FRAME = "CONSTS.MEAS.GET_FRAME"
CONSTS.MEAS.SAVE_SWEEP = "CONSTS.MEAS.SAVE_SWEEP"
CONSTS.MEAS.SAVE_SWEEP_W_FIT = "CONSTS.MEAS.SAVE_SWEEP_W_FIT"
CONSTS.MEAS.SAVE_FULL_DATA = "CONSTS.MEAS.SAVE_FULL_DATA"
CONSTS.MEAS.ADD = "CONSTS.MEAS.ADD"
CONSTS.MEAS.START = "CONSTS.MEAS.START"
CONSTS.MEAS.STOP = "CONSTS.MEAS.STOP"
CONSTS.MEAS.PAUSE = "CONSTS.MEAS.PAUSE"
CONSTS.MEAS.CLOSE = "CONSTS.MEAS.CLOSE"
CONSTS.MEAS.SET_AOI = "CONSTS.MEAS.SET_AOI"
CONSTS.MEAS.SET_FRAME_NUM = "CONSTS.MEAS.SET_FRAME_NUM"
CONSTS.MEAS.SET_ROLLING_AVG_WINDOW = "CONSTS.MEAS.SET_ROLLING_AVG_WINDOW"
CONSTS.MEAS.SET_ROLLING_AVG_MAX_SWEEPS = "CONSTS.MEAS.SET_ROLLING_AVG_MAX_SWEEPS"

# Camera commands
CONSTS.CAM = types.SimpleNamespace()
CONSTS.CAM.SET_CAMERA_PARAMS = "CONSTS.CAM.SET_CAMERA_PARAMS"
CONSTS.CAM.TAKE_SNAPSHOT = "CONSTS.CAM.TAKE_SNAPSHOT"
CONSTS.CAM.TAKE_AND_SAVE_SNAPSHOT = "CONSTS.CAM.TAKE_AND_SAVE_SNAPSHOT"
CONSTS.CAM.GET_FRAME_SHAPE = "CONSTS.CAM.GET_FRAME_SHAPE"
CONSTS.CAM.START_VIDEO = "CONSTS.CAM.START_VIDEO"
CONSTS.CAM.STOP_VIDEO = "CONSTS.CAM.STOP_VIDEO"

# NOTE are RF commands required?? Always set through SEQGEN cmds below?
# RF commands
CONSTS.RF = types.SimpleNamespace()
CONSTS.RF.SET_PARAMS = "CONSTS.RF.SET_PARAMS"

# Signal generator commands
CONSTS.SEQGEN = types.SimpleNamespace()
CONSTS.SEQGEN.LASER_RF_OUTPUT = "CONSTS.SEQGEN.LASER_RF_OUTPUT"
CONSTS.SEQGEN.LASER_OUTPUT = "CONSTS.SEQGEN.LASER_OUTPUT"
CONSTS.SEQGEN.RF_OUTPUT = "CONSTS.SEQGEN.RF_OUTPUT"
