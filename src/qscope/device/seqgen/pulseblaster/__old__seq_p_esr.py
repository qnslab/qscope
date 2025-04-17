# This function defines the ODMR sequence for the pulseblaster
from __future__ import annotations

import typing

from loguru import logger

if typing.TYPE_CHECKING:
    from .pulseblaster import PulseBlaster


def seq_p_esr(
    pb: PulseBlaster,
    sequence_params: dict[str, float],
    laser_dur: float = 3e-3,
    rf_dur: float = 100e-9,
    laser_delay: float = 0,
    rf_delay: float = 0,
    ref_mode: str = "no_rf",
    exp_t: float = 30e-3,
    sweep_len: int = 1,
    camera_trig_time: float = 0,
    **kwargs,
):
    if ref_mode == ("" or None):
        b_ref = False
        logger.info("Setting up pulsed ESR sequence with no reference")
    else:
        b_ref = True
        logger.info("Setting up pulsed ESR sequence with {} as a reference", ref_mode)

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
    rf_dur = rf_dur * 1e9
    laser_delay = laser_delay * 1e9
    rf_delay = rf_delay * 1e9
    exp_t = exp_t * 1e9
    trigger_time = trigger_time * 1e9

    # Get the number of cycles for the inner loops
    cycle_time = laser_delay + rf_delay + laser_dur + rf_dur
    trigger_loops = int(trigger_time / cycle_time) + 1
    num_loops = int(exp_t / cycle_time) + 1

    # Start the programming of the pulseblaster
    pb.start_programming()

    # Turn the laser on to initialize the system
    pb.add_pulse(["laser"], exp_t)

    for i in range(0, sweep_len):
        # signal pulse sequence with camera on
        seq_sig(
            laser_dur,
            rf_dur,
            laser_delay,
            rf_delay,
            num_loops,
            pb,
            const_trigger=["camera"],
        )

        # Camera readout time with the same sequence running

        if ref_mode == "f_mod":
            seq_sig(
                laser_dur,
                rf_dur,
                laser_delay,
                rf_delay,
                trigger_loops,
                pb,
                const_trigger=["rf_trig"],
            )

        else:
            seq_sig(laser_dur, rf_dur, laser_delay, rf_delay, trigger_loops, pb)

        # Reference pulse sequence
        if b_ref:
            if ref_mode == "no_rf":
                seq_ref_no_ref(
                    laser_dur,
                    rf_dur,
                    laser_delay,
                    rf_delay,
                    num_loops,
                    pb,
                    const_trigger=["camera"],
                )

                # Trigger the signal generator to step to the next frequency
                pb.add_pulse(["laser", "mw_x", "rf_trig"], trigger_time)
            elif ref_mode == "f_mod":
                seq_sig(laser_dur, rf_dur, laser_delay, rf_delay, trigger_loops, pb)

                seq_sig(
                    laser_dur,
                    rf_dur,
                    laser_delay,
                    rf_delay,
                    trigger_loops,
                    pb,
                    const_trigger=["rf_trig"],
                )

    pb.add_pulse([], trigger_time)

    # Turn the laser off and end sequence
    pb.end_sequence(10e6)

    # End of pulse program
    pb.stop_programming()


###########################################################################
#                            Signal sequence                              #
###########################################################################


def seq_sig(laser_dur, rf_dur, laser_delay, rf_delay, num_loops, pb, const_trigger=[]):
    # if const_trigger == []:
    #     inst_sig = pb.add_pulse(
    #         ["laser"], laser_dur, delay=laser_delay, loop="start", num=num_loops
    #     )

    #     pb.add_pulse(["mw_x"], rf_dur, delay=rf_delay, loop="end", inst=inst_sig)
    # else:
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


def seq_ref_no_ref(
    laser_dur, rf_dur, laser_delay, rf_delay, num_loops, pb, const_trigger=[]
):
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
