# This function defines the ODMR sequence for the pulseblaster
from __future__ import annotations

import typing

import numpy as np
from loguru import logger

if typing.TYPE_CHECKING:
    from .pulseblaster import PulseBlaster

from qscope.device.seqgen.pulse_kernel import PulseKernel


def seq_rabi(
    seqgen: PulseBlaster,
    sequence_params: dict[str, float],
    sweep_x: np.ndarray = None,
    laser_dur: float = 3e-6,
    laser_delay: float = 0,
    laser_to_rf_delay: float = 200e-9,
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
    laser_to_rf_delay = int(laser_to_rf_delay * 1e9)
    rf_delay = int(rf_delay * 1e9)
    exp_t = int(exp_t * 1e9)
    trigger_time = int(trigger_time * 1e9)

    # time_list = np.linspace(time_start, time_stop, time_num)
    time_list = 1e9 * sweep_x
    # Make the time list a list of integers
    time_list = [int(i) for i in time_list.tolist()]

    # --- SIGNAL KERNEL ---
    # initialise the pulse series object
    pk_sig = PulseKernel(seqgen.ch_defs)
    # Program the kernel pulses
    pk_sig.add_pulse(["laser"], 0, laser_dur, ch_delay=laser_delay)
    pk_sig.append_delay(laser_to_rf_delay)
    pk_sig.append_pulse(["rf_x"], 12, ch_delay=rf_delay, var_dur=True)
    pk_sig.finish_kernel()

    # --- REFERENCE KERNEL ---
    # Program the kernel pulses
    pk_ref = PulseKernel(seqgen.ch_defs)
    # Program the kernel pulses
    pk_ref.add_pulse(["laser"], 0, laser_dur, ch_delay=laser_delay)
    pk_ref.append_delay(laser_to_rf_delay)
    pk_ref.append_delay(12, var_dur=True)
    pk_ref.finish_kernel()

    # Get the base kernel time
    base_time = pk_sig.get_end_time() - 12

    # Get the number of cycles for the inner loops
    num_loops = int(exp_t / base_time)
    trigger_loops = int(trigger_time / base_time)
    # Add 5% to the number of loops to make sure the sequence is long enough
    trigger_loops = int(trigger_loops * 1.05)

    logger.info(
        "Programming Ramsey sequence with the following parameters:"
        + f"\nLaser duration: {laser_dur} ns"
        + f"\nFirst RF pulse duration: {time_list[0]} ns"
        + f"\nLast RF pulse duration: {time_list[-1]} ns"
        + f"\nLaser delay: {laser_delay} ns"
        + f"\nRF delay: {rf_delay} ns"
        + f"\nReference mode: {ref_mode}"
        + f"\nBase time: {base_time} ns"
        + f"\nCamera exposure time: {exp_t * 1e-6} ms"
        + f"\nNumber of loops: {num_loops}"
        + f"\nCamera trigger time: {trigger_time * 1e-6} ms"
        + f"\nNumber of trigger loops: {trigger_loops}"
    )

    # --- PROGRAMMING ---
    seqgen.start_programming()
    # Initial laser pulse
    seqgen.add_instruction(**{"active_chs": ["laser"], "dur": 2 * exp_t})

    for tau in time_list:
        if avg_per_point > 1:
            # Make a trigger loop for the averaging that is as short as possible
            inst = seqgen.add_instruction([], 12, loop="start", num=avg_per_point)
        # update the time in the kernel
        pk_sig.update_var_durs(tau)

        # Add the SIG kernel to the sequence generator
        seqgen.add_kernel(pk_sig, num_loops, const_chs=["camera"])

        # Recalculate the trigger loops to maintain the same trigger time.
        # base_time = pk_sig.get_end_time()
        # # trigger_loops = int(trigger_time / base_time)
        # # trigger_loops = int(trigger_loops * 1.1)
        # logger.info( f"\ntau: {tau} ns"
        #     + f"\nBase time: {base_time} ns")
        seqgen.add_kernel(pk_sig, trigger_loops, const_chs=[])

        # Reference pulse sequence
        if b_ref:
            # update the time in the kernel
            pk_ref.update_var_durs(tau)

            # Add the REF kernel to the sequence generator
            seqgen.add_kernel(pk_ref, num_loops, const_chs=["camera"])
            seqgen.add_kernel(pk_ref, trigger_loops, const_chs=[])

            if avg_per_point > 1:
                seqgen.add_instruction([], 12, loop="end", inst=inst)

    # seqgen.add_instruction([], trigger_time)

    # Turn the laser off and end sequence
    seqgen.end_sequence(1e6)

    # End of pulse program
    seqgen.stop_programming()

    return pk_sig, pk_ref


###########################################################################
#                            Signal sequence                              #
###########################################################################


def seq_sig(
    laser_dur, rf_dur, laser_delay, rf_delay, num_loops, seqgen, const_trigger=[]
):
    if rf_dur == 0:
        seqgen.add_pulse(
            ["laser"],
            laser_dur * num_loops,
            delay=laser_delay,
            const_trigger=const_trigger,
        )
    else:
        inst_sig = seqgen.add_pulse(
            ["laser"],
            laser_dur,
            delay=laser_delay,
            loop="start",
            num=num_loops,
            const_trigger=const_trigger,
        )

        seqgen.add_pulse(
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
    laser_dur, rf_dur, laser_delay, rf_delay, num_loops, seqgen, const_trigger=[]
):
    if rf_dur == 0:
        seqgen.add_pulse(
            ["laser"],
            laser_dur * num_loops,
            delay=laser_delay,
            const_trigger=const_trigger,
        )
    else:
        inst_sig = seqgen.add_pulse(
            ["laser"],
            laser_dur,
            delay=laser_delay,
            loop="start",
            num=num_loops,
            const_trigger=const_trigger,
        )

        seqgen.add_pulse(
            [],
            rf_dur,
            delay=rf_delay,
            loop="end",
            inst=inst_sig,
            const_trigger=const_trigger,
        )
