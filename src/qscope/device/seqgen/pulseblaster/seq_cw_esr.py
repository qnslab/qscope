# This function defines the ODMR sequence for the pulseblaster
from __future__ import annotations

import copy
import typing

from loguru import logger

from qscope.device.seqgen.pulse_kernel import PulseKernel

if typing.TYPE_CHECKING:
    from .pulseblaster import PulseBlaster


def seq_cw_esr(
    seqgen: PulseBlaster,
    sequence_params: dict[str, float],
    ref_mode: str = "no_rf",
    exp_t: float = 30e-6,
    sweep_len: int = 1,
    avg_per_point: int = 1,
    camera_trig_time: float = 0,
    **kwargs,
):
    if camera_trig_time == 0:
        trigger_time = sequence_params["camera_trig_time"]  # s
    else:
        trigger_time = camera_trig_time
    if ref_mode == ("" or None):
        b_ref = False
        logger.info("Setting up CW ESR sequence with no reference")
    else:
        b_ref = True
        logger.info("Setting up CW ESR sequence with {} as a reference", ref_mode)

    # convert times to ns assuming the user inputs in s
    exp_t *= 1e9  # s to ns
    trigger_time *= 1e9  # s to ns

    # ------- SIGNAL -------
    pk_sig = PulseKernel(seqgen.ch_defs)
    # Program the kernel pulses
    pk_sig.add_pulse(["laser", "rf_x"], 0, exp_t)

    # ------- Trigger -------
    pk_sig_trig = PulseKernel(seqgen.ch_defs)
    pk_sig_trig.add_pulse(["laser", "rf_x"], 0, trigger_time)

    # ------- REFERENCE -------
    pk_ref = PulseKernel(seqgen.ch_defs)
    if ref_mode == "no_rf":
        pk_ref.append_pulse(["laser"], exp_t)
    elif ref_mode == "no_laser":
        pk_ref.append_pulse(["rf_x"], exp_t)
    elif ref_mode == "fmod":
        pk_ref.append_pulse(["laser", "rf_x"], exp_t)

    # ------- Trigger -------
    pk_ref_trig = PulseKernel(seqgen.ch_defs)
    if ref_mode == "no_rf":
        pk_ref_trig.append_pulse(["laser"], trigger_time)
    elif ref_mode == "no_laser":
        pk_ref_trig.append_pulse(["rf_x"], trigger_time)
    elif ref_mode == "fmod":
        pk_ref_trig.append_pulse(["laser", "rf_x"], trigger_time)

    # Start the programming of the pulseblaster
    seqgen.start_programming()

    # Turn the laser on to initialize the system
    seqgen.add_instruction(**{"active_chs": ["laser"], "dur": exp_t})

    for i in range(0, sweep_len):
        # Start the per point average loop if required
        if avg_per_point > 1:
            inst = seqgen.add_instruction([], 12, loop="start", num=avg_per_point)

        # Add the SIG kernel to the sequence generator
        seqgen.add_kernel(pk_sig, 1, const_chs=["camera"])
        if ref_mode == "fmod":
            # trigger the segnal generator to go to the next freq
            seqgen.add_kernel(pk_sig_trig, 1, const_chs=["rf_trig"])
        else:
            seqgen.add_kernel(pk_sig_trig, 1)

        if b_ref:
            seqgen.add_kernel(pk_ref, 1, const_chs=["camera"])
            seqgen.add_kernel(pk_ref_trig, 1, const_chs=["rf_trig"])

        if avg_per_point > 1:
            seqgen.add_instruction([], 12, loop="end", inst=inst)

        seqgen.add_instruction([], dur=trigger_time)

    # Turn the laser off and end sequence
    seqgen.end_sequence(10e6)

    # End of pulse program
    seqgen.stop_programming()

    return pk_sig, pk_ref
