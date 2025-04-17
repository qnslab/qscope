# This function defines the ODMR sequence for the pulseblaster
from __future__ import annotations

import typing

from loguru import logger

if typing.TYPE_CHECKING:
    from .pulseblaster import PulseBlaster

from qscope.device.seqgen.pulse_kernel import PulseKernel


def seq_p_esr(
    seqgen: PulseBlaster,
    sequence_params: dict[str, float],
    laser_dur: float = 3e-3,
    rf_dur: float = 100e-9,
    laser_delay: float = 0,
    laser_to_rf_delay: float = 300e-9,
    rf_delay: float = 0,
    ref_mode: str = "no_rf",
    exp_t: float = 30e-3,
    sweep_len: int = 1,
    avg_per_point: int = 1,
    camera_trig_time: float = 0,
    **kwargs,
):
    if ref_mode == ("" or None):
        b_ref = False
        logger.info("Setting up pulsed ESR sequence with no reference")
    else:
        b_ref = True
        logger.info("Setting up pulsed ESR sequence with {} as a reference", ref_mode)

    if camera_trig_time == 0:
        trigger_time = sequence_params["camera_trig_time"]  # s
    else:
        trigger_time = camera_trig_time

    # convert times to ns assuming the user inputs in s
    laser_dur = int(laser_dur * 1e9)
    rf_dur = int(rf_dur * 1e9)
    laser_delay = int(laser_delay * 1e9)
    laser_to_rf_delay = int(laser_to_rf_delay * 1e9)
    rf_delay = int(rf_delay * 1e9)
    exp_t = int(exp_t * 1e9)
    trigger_time = int(trigger_time * 1e9)

    # Create the SIGNAL kernel
    pk_sig = PulseKernel(seqgen.ch_defs)
    # Program the kernel pulses
    pk_sig.add_pulse(["laser"], 0, laser_dur, ch_delay=laser_delay)
    pk_sig.append_delay(laser_to_rf_delay)
    pk_sig.append_pulse(["rf_x"], rf_dur, ch_delay=rf_delay)
    pk_sig.finish_kernel()

    # create the REFERENCE kernel
    pk_ref = PulseKernel(seqgen.ch_defs)
    # Program the kernel pulses
    pk_ref.add_pulse(["laser"], 0, laser_dur, ch_delay=laser_delay)
    pk_ref.append_delay(laser_to_rf_delay)
    pk_ref.append_delay(rf_dur)
    pk_ref.finish_kernel()

    # Get the base kernel time
    base_time = pk_sig.get_end_time()

    # Get the number of cycles for the inner loops
    trigger_loops = int(trigger_time / base_time) + 1
    num_loops = int(exp_t / base_time) + 1

    logger.info(
        "Programming Ramsey sequence with the following parameters:"
        + f"\nLaser duration: {laser_dur} ns"
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

    # Turn the laser on to initialize the system
    seqgen.add_instruction(**{"active_chs": ["laser"], "dur": 2 * exp_t})

    for i in range(0, sweep_len):
        # Start the per point average loop if required
        if avg_per_point > 1:
            inst = seqgen.add_instruction([], 12, loop="start", num=avg_per_point)

        # Add the SIG kernel to the sequence generator
        seqgen.add_kernel(pk_sig, num_loops, const_chs=["camera"])
        if ref_mode == "f_mod":
            # trigger the segnal generator to go to the next freq
            seqgen.add_kernel(pk_sig, trigger_loops, const_chs=["rf_trig"])
        else:
            seqgen.add_kernel(pk_sig, trigger_loops)

        # Reference pulse sequence
        if b_ref:
            seqgen.add_kernel(pk_ref, num_loops, const_chs=["camera"])
            seqgen.add_kernel(pk_ref, trigger_loops, const_chs=["rf_trig"])

        if avg_per_point > 1:
            seqgen.add_instruction([], 12, loop="end", inst=inst)

    seqgen.add_instruction([], dur=trigger_time)

    # Turn the laser off and end sequence
    seqgen.end_sequence(1e6)

    # End of pulse program
    seqgen.stop_programming()

    return pk_sig, pk_ref
