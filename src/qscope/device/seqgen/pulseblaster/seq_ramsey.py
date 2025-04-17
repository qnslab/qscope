# This function defines the ODMR sequence for the pulseblaster
from __future__ import annotations

import typing

import numpy as np
from loguru import logger

from qscope.device.seqgen.pulse_kernel import PulseKernel

if typing.TYPE_CHECKING:
    from .pulseblaster import PulseBlaster


def seq_ramsey(
    seqgen: PulseBlaster,
    sequence_params: dict[str, float],
    sweep_x: np.ndarray = None,
    pi_2_dur: float = 100e-9,
    laser_dur: float = 3e-3,
    laser_delay: float = 0,
    laser_to_rf_delay: float = 300e-9,
    rf_delay: float = 0,
    ref_mode: str = "",
    exp_t: float = 10e-3,
    avg_per_point: int = 1,
    camera_trig_time: float = 0,
    **kwargs,
):
    if ref_mode == ("" or None):
        b_ref = False
        logger.info("Setting up Ramsey sequence with no reference")
    else:
        b_ref = True
        logger.info("Setting up Ramsey sequence with {} as a reference", ref_mode)

    if camera_trig_time == 0:
        trigger_time = sequence_params["camera_trig_time"]  # s
    else:
        trigger_time = camera_trig_time
    # convert times to ns assuming the user inputs in s
    laser_dur = int(np.ceil(laser_dur * 1e9))
    laser_to_rf_delay = int(np.ceil(laser_to_rf_delay * 1e9))
    pi_2_dur = int(np.ceil(pi_2_dur * 1e9))
    laser_delay = int(np.ceil(laser_delay * 1e9))
    rf_delay = int(np.ceil(rf_delay * 1e9))
    exp_t = int(np.ceil(exp_t * 1e9))
    trigger_time = int(np.ceil(trigger_time * 1e9))

    # time_list = np.linspace(time_start, time_stop, time_num)
    time_list = np.ceil(1e9 * sweep_x)
    # Make the time list a list of integers
    time_list = [int(i) for i in time_list.tolist()]

    # --- SIGNAL KERNEL ---
    # initialise the pulse series object
    pk_sig = PulseKernel(seqgen.ch_defs)
    # Program the kernel pulses
    pk_sig.add_pulse(["laser"], 0, laser_dur, ch_delay=laser_delay)
    pk_sig.append_delay(laser_to_rf_delay)
    pk_sig.append_pulse(["rf_x"], pi_2_dur, ch_delay=rf_delay)
    pk_sig.append_delay(2, var_dur=True)
    pk_sig.append_pulse(["rf_x"], pi_2_dur, ch_delay=rf_delay)
    pk_sig.finish_kernel()

    # --- REFERENCE KERNEL ---
    # Program the kernel pulses
    pk_ref = PulseKernel(seqgen.ch_defs)
    # Program the kernel pulses
    pk_ref.add_pulse(["laser"], 0, laser_dur, ch_delay=laser_delay)
    pk_ref.append_delay(laser_to_rf_delay)
    pk_ref.append_pulse(["rf_x"], pi_2_dur, ch_delay=rf_delay)
    pk_ref.append_delay(2, var_dur=True)

    if ref_mode == "-π/2 at end":
        pk_ref.append_pulse(["rf_-x"], pi_2_dur, ch_delay=rf_delay)
    elif ref_mode == "3π/2 at end":
        pk_ref.append_pulse(["rf_x"], 3 * pi_2_dur, ch_delay=rf_delay)
    else:
        pk_ref.append_pulse(["rf_-x"], pi_2_dur, ch_delay=rf_delay)
    pk_ref.finish_kernel()

    # Get the base kernel time
    base_time = pk_sig.get_end_time() - 12

    # Get the number of cycles for the inner loops
    num_loops = int(np.ceil(exp_t / base_time))
    trigger_loops = int(np.ceil(trigger_time / base_time))
    # Add 5% to the number of loops to make sure the sequence is long enough
    trigger_loops = int(np.ceil(trigger_loops * 1.05))

    logger.info(
        "Programming Ramsey sequence with the following parameters:"
        + f"\nTime start: {time_list[0]}"
        + f"\nTime stop: {time_list[-1]}"
        + f"\nTime num: {len(time_list)}"
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

    # Start the programming of the pulseblaster
    seqgen.start_programming()
    # Initial laser pulse
    seqgen.add_instruction(**{"active_chs": ["laser"], "dur": exp_t})

    for tau in time_list:
        if avg_per_point > 1:
            # Make a trigger loop for the averaging that is as short as possible
            inst = seqgen.add_instruction(
                [], 12, loop="start", num=avg_per_point, const_chs=["camera"]
            )

        pk_sig.update_var_durs(tau)

        # Add the SIG kernel to the sequence generator
        seqgen.add_kernel(pk_sig, num_loops, const_chs=["camera"])
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

    seqgen.add_instruction([], trigger_time)

    # Turn the laser off and end sequence
    seqgen.end_sequence(10e6)

    # End of pulse program
    seqgen.stop_programming()

    return pk_sig, pk_ref
