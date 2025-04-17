# This function defines the ODMR sequence for the pulseblaster
from __future__ import annotations

import typing

import numpy as np
from loguru import logger

if typing.TYPE_CHECKING:
    from .pulseblaster import PulseBlaster


def seq_ramsey(
    pb: PulseBlaster,
    sequence_params: dict[str, float],
    sweep_x: np.ndarray = None,
    laser_dur: float = 3e-3,
    pi_2_dur: float = 100e-9,
    laser_delay: float = 0,
    rf_delay: float = 0,
    ref_mode: str = "",
    exp_t: float = 10e-3,
    camera_trig_time: float = 0,
    **kwargs,
):
    if ref_mode == ("" or None):
        b_ref = False
        logger.info("Setting up Ramsey sequence with no reference")
    else:
        b_ref = True
        logger.info("Setting up Ramsey sequence with {} as a reference", ref_mode)

    if laser_delay == 0:
        laser_delay = sequence_params["laser_delay"]
    if rf_delay == 0:
        rf_delay = sequence_params["rf_delay"]

    if camera_trig_time == 0:
        trigger_time = sequence_params["camera_trig_time"]  # s
    else:
        trigger_time = camera_trig_time
    # convert times to ns assuming the user inputs in s
    laser_dur = laser_dur * 1e9
    pi_2_dur = pi_2_dur * 1e9
    laser_delay = laser_delay * 1e9
    rf_delay = rf_delay * 1e9
    exp_t = exp_t * 1e9
    trigger_time = trigger_time * 1e9

    # time_list = np.linspace(time_start, time_stop, time_num)
    time_list = 1e9 * sweep_x
    # Make the time list a list of integers
    time_list = [int(i) for i in time_list.tolist()]

    # Get the number of cycles for the inner loops
    cycle_time = laser_delay + laser_dur + time_list[0]
    trigger_loops = int(trigger_time / cycle_time) + 1
    num_loops = int(exp_t / cycle_time) + 1

    logger.info(
        "Programming Ramsey sequence with the following parameters:"
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

    # Start the programming of the pulseblaster
    pb.start_programming()

    # Turn the laser on to initialize the system
    pb.add_pulse(["laser"], trigger_time)
    pb.add_pulse(["laser"], exp_t)

    for tau in time_list:
        # signal pulse sequence with camera on
        seq_sig(
            pb,
            laser_dur=laser_dur,
            tau=tau,
            laser_delay=laser_delay,
            pi_2_dur=pi_2_dur,
            rf_delay=rf_delay,
            num_loops=num_loops,
            const_trigger=["camera"],
        )

        # Camera readout time with the same sequence running
        seq_sig(
            pb,
            laser_dur=laser_dur,
            tau=tau,
            pi_2_dur=pi_2_dur,
            rf_delay=rf_delay,
            laser_delay=laser_delay,
            num_loops=trigger_loops,
        )

        # Reference pulse sequence
        if b_ref:
            if ref_mode == "-π/2 at end":
                ref_sig_minus_pi_2_at_end(
                    pb,
                    laser_dur=laser_dur,
                    tau=tau,
                    pi_2_dur=pi_2_dur,
                    laser_delay=laser_delay,
                    rf_delay=rf_delay,
                    num_loops=num_loops,
                    const_trigger=["camera"],
                )
                # Camera readout time with the same sequence running
                ref_sig_minus_pi_2_at_end(
                    pb,
                    laser_dur=laser_dur,
                    tau=tau,
                    pi_2_dur=pi_2_dur,
                    laser_delay=laser_delay,
                    rf_delay=rf_delay,
                    num_loops=trigger_loops,
                )
            elif ref_mode == "3π/2 at end":
                ref_sig_3pi_2_at_end(
                    pb,
                    laser_dur=laser_dur,
                    tau=tau,
                    pi_2_dur=pi_2_dur,
                    laser_delay=laser_delay,
                    rf_delay=rf_delay,
                    num_loops=num_loops,
                    const_trigger=["camera"],
                )
                ref_sig_3pi_2_at_end(
                    pb,
                    laser_dur=laser_dur,
                    tau=tau,
                    pi_2_dur=pi_2_dur,
                    laser_delay=laser_delay,
                    rf_delay=rf_delay,
                    num_loops=trigger_loops,
                )

    pb.add_pulse([], trigger_time)

    # Turn the laser off and end sequence
    pb.end_sequence(10e6)

    # End of pulse program
    pb.stop_programming()


###########################################################################
#                            Signal sequence                              #
###########################################################################


def seq_sig(
    pb: PulseBlaster,
    laser_dur: int = 0,
    tau: int = 0,
    pi_2_dur: int = 0,
    laser_delay: int = 0,
    rf_delay: int = 0,
    num_loops: int = 0,
    const_trigger: list = [],
):
    # Add the laser pulse
    inst_sig = pb.add_pulse(
        ["laser"],
        laser_dur,
        delay=laser_delay,
        loop="start",
        num=num_loops,
        const_trigger=const_trigger,
    )

    # Add pi/2 pulse
    pb.add_pulse(["mw_x"], pi_2_dur, delay=rf_delay, const_trigger=const_trigger)

    # Add the dark time
    pb.add_pulse([], tau, const_trigger=const_trigger)

    # Add pi/2 pulse
    pb.add_pulse(
        ["mw_x"],
        pi_2_dur,
        delay=rf_delay,
        loop="end",
        inst=inst_sig,
        const_trigger=const_trigger,
    )


###########################################################################
#                          Reference sequences                            #
###########################################################################


def ref_sig_minus_pi_2_at_end(
    pb: PulseBlaster,
    laser_dur: int = 0,
    tau: int = 0,
    pi_2_dur: int = 0,
    laser_delay: int = 0,
    rf_delay: int = 0,
    num_loops: int = 0,
    const_trigger: list = [],
):
    # Add the laser pulse
    inst_sig = pb.add_pulse(
        ["laser"],
        laser_dur,
        delay=laser_delay,
        loop="start",
        num=num_loops,
        const_trigger=const_trigger,
    )

    # Add pi/2 pulse
    pb.add_pulse(["mw_x"], pi_2_dur, delay=rf_delay, const_trigger=const_trigger)

    # Add the dark time
    pb.add_pulse([], tau, const_trigger=const_trigger)

    # Add pi/2 pulse
    pb.add_pulse(
        ["mw_-x"],
        pi_2_dur,
        delay=rf_delay,
        loop="end",
        inst=inst_sig,
        const_trigger=const_trigger,
    )


def ref_sig_3pi_2_at_end(
    pb: PulseBlaster,
    laser_dur: int = 0,
    tau: int = 0,
    pi_2_dur: int = 0,
    laser_delay: int = 0,
    rf_delay: int = 0,
    num_loops: int = 0,
    const_trigger: list = [],
):
    # Add the laser pulse
    inst_sig = pb.add_pulse(
        ["laser"],
        laser_dur,
        delay=laser_delay,
        loop="start",
        num=num_loops,
        const_trigger=const_trigger,
    )

    # Add pi/2 pulse
    pb.add_pulse(["mw_x"], pi_2_dur, delay=rf_delay, const_trigger=const_trigger)

    # Add the dark time
    pb.add_pulse([], tau, const_trigger=const_trigger)

    # Add pi/2 pulse
    pb.add_pulse(
        ["mw_x"],
        3 * pi_2_dur,
        delay=rf_delay,
        loop="end",
        inst=inst_sig,
        const_trigger=const_trigger,
    )
