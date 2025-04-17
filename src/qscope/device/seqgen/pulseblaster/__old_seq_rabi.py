# This function defines the ODMR sequence for the pulseblaster
from __future__ import annotations

import typing

import numpy as np
from loguru import logger

if typing.TYPE_CHECKING:
    from .pulseblaster import PulseBlaster


def seq_rabi(
    pb: PulseBlaster,
    sequence_params: dict[str, float],
    sweep_x: np.ndarray = None,
    laser_dur: float = 3e-3,
    laser_delay: float = 0,
    rf_delay: float = 0,
    ref_mode: str = "no_rf",
    exp_t: float = 30e-3,
    avg_per_point: int = 1,
    camera_trig_time: float = 0,
    **kwargs,
):
    if ref_mode == ("" or None):
        b_ref = False
        logger.info("Setting up Rabi sequence with no reference")
    else:
        b_ref = True
        logger.info("Setting up Rabi sequence with {} as a reference", ref_mode)

    if laser_delay == 0:
        laser_delay = sequence_params["laser_delay"]
    if rf_delay == 0:
        rf_delay = sequence_params["rf_delay"]

    if camera_trig_time == 0:
        trigger_time = sequence_params["camera_trig_time"]  # s
    else:
        trigger_time = camera_trig_time
    # convert times to ns assuming the user inputs in s
    laser_dur = int(laser_dur * 1e9)
    laser_delay = int(laser_delay * 1e9)
    rf_delay = int(rf_delay * 1e9)
    exp_t = int(exp_t * 1e9)
    trigger_time = int(trigger_time * 1e9)

    # time_list = np.linspace(time_start, time_stop, time_num)
    time_list = 1e9 * sweep_x
    # Make the time list a list of integers
    time_list = [int(i) for i in time_list.tolist()]

    logger.info(
        "Programming RABI sequence with the following parameters:"
        + f"\nTime start: {time_list[0]}"
        + f"\nTime stop: {time_list[-1]}"
        + f"\nTime num: {len(time_list)}"
        + f"\nLaser duration: {laser_dur}"
        + f"\nLaser delay: {laser_delay}"
        + f"\nRF delay: {rf_delay}"
        + f"\nReference mode: {ref_mode}"
        + f"\nExperiment time: {exp_t * 1e-6} ms"
        + f"\nCamera trigger time: {trigger_time * 1e-6} ms"
    )

    # Get the number of cycles for the inner loops
    cycle_time = laser_delay + rf_delay + laser_dur + time_list[0]
    trigger_loops = int(trigger_time / cycle_time) + 1
    num_loops = int(exp_t / cycle_time) + 1

    # Start the programming of the pulseblaster
    pb.start_programming()

    # Turn the laser on to initialize the system
    pb.add_pulse(["laser"], trigger_time)
    pb.add_pulse(["laser"], exp_t)

    for tau in time_list:
        if avg_per_point > 1:
            # Make a trigger loop for the averaging that is as short as possible
            avg_inst = pb.add_pulse(["laser"], 12, loop="start")

        # signal pulse sequence with camera on
        seq_sig(
            laser_dur,
            tau,
            laser_delay,
            rf_delay,
            num_loops,
            pb,
            const_trigger=["camera"],
        )

        # Camera readout time with the same sequence running
        seq_sig(laser_dur, tau, laser_delay, rf_delay, trigger_loops, pb)

        # Reference pulse sequence
        if b_ref:
            if ref_mode == "no_rf":
                seq_ref_no_rf(
                    laser_dur,
                    tau,
                    laser_delay,
                    rf_delay,
                    num_loops,
                    pb,
                    const_trigger=["camera"],
                )

                seq_ref_no_rf(laser_dur, tau, laser_delay, rf_delay, trigger_loops, pb)

            if avg_per_point > 1:
                # add the end of the avg loop (short as possible)
                pb.add_pulse(["laser"], 12, loop="end", inst=avg_inst)

        # Trigger the signal generator to step to the next frequency
        # pb.add_pulse(["laser", "mw_x", "rf_trig"], trigger_time)

    pb.add_pulse([], trigger_time)

    # Turn the laser off and end sequence
    pb.end_sequence(10e6)

    # End of pulse program
    pb.stop_programming()


###########################################################################
#                            Signal sequence                              #
###########################################################################


def seq_sig(laser_dur, rf_dur, laser_delay, rf_delay, num_loops, pb, const_trigger=[]):
    if rf_dur == 0:
        pb.add_pulse(
            ["laser"],
            laser_dur * num_loops,
            delay=laser_delay,
            const_trigger=const_trigger,
        )
    else:
        inst_sig = pb.add_pulse(
            ["laser"],
            laser_dur,
            delay=laser_delay,
            loop="start",
            num=num_loops,
            const_trigger=const_trigger,
        )

        pb.add_pulse(
            ["mw_x"],
            rf_dur,
            delay=rf_delay,
            loop="end",
            inst=inst_sig,
            const_trigger=const_trigger,
        )


###########################################################################
#                          Reference sequences                            #
###########################################################################


def seq_ref_no_rf(
    laser_dur, rf_dur, laser_delay, rf_delay, num_loops, pb, const_trigger=[]
):
    if rf_dur == 0:
        pb.add_pulse(
            ["laser"],
            laser_dur * num_loops,
            delay=laser_delay,
            const_trigger=const_trigger,
        )
    else:
        inst_sig = pb.add_pulse(
            ["laser"],
            laser_dur,
            delay=laser_delay,
            loop="start",
            num=num_loops,
            const_trigger=const_trigger,
        )

        pb.add_pulse(
            [],
            rf_dur,
            delay=rf_delay,
            loop="end",
            inst=inst_sig,
            const_trigger=const_trigger,
        )
