# utility functions for pulseblaster
# Wrapper for the pulseblaster communication

from __future__ import annotations

import typing

import numpy as np
from loguru import logger

import qscope.device.seqgen.pulseblaster.spinapi as pb
from qscope.device import Device
from qscope.device.seqgen.pulseblaster.spinapi import *
from qscope.util import format_error_response

from .seq_cw_esr_long_exposure import seq_cw_esr_long_exposure
from .seq_cw_esr import seq_cw_esr
from .seq_p_esr import seq_p_esr
from .seq_rabi import seq_rabi
from .seq_ramsey import seq_ramsey
from .seq_spin_echo import seq_spin_echo
from .seq_t1 import seq_t1


class PulseBlaster(Device):
    connected: bool = False
    board_num: str
    ch_defs: dict[str, str]
    sequence_params: dict[str, float]
    required_config: dict[str, type] = {
        "board_num": str,
        "ch_defs": dict,  # dict[str, str]
        "sequence_params": dict,  # dict[str, float]
    }

    def __init__(
        self,
        board_num: str,
        ch_defs: dict[str, str],
        sequence_params: dict[str, float],
    ):
        super().__init__(
            board_num=board_num,
            ch_defs=ch_defs,
            sequence_params=sequence_params,
        )
        # Define the control chs of the system
        self.shortest_dur = int(12)  # ns

    def unroll_metadata(self):
        # override normal unroll_metadata as we don't use underscores... could be cleaned up.
        return {
            "board_num": self.board_num,
            "ch_defs": self.ch_defs,
            "sequence_params": self.sequence_params,
        }

    def open(self) -> tuple[bool, str]:
        if pb.pb_count_boards() < 0:
            logger.error("No Pulseblaster boards found")
            self.connected = False
            return False, "Error: No Pulseblaster boards found"
        try:
            pb.pb_select_board(int(self.board_num))
            pb.pb_init()
            pb.pb_core_clock(500)  # in MHz
            logger.info("Pulseblaster opened, status: {}", pb.pb_read_status())
            self.connected = True
            return True, "Pulseblaster opened"
        except Exception as e:
            logger.exception("Error opening Pulseblaster: {}", format_error_response())
            return False, f"Error opening Pulseblaster: {format_error_response()}"

    def close(self):
        if self.connected:
            self.stop()
            pb.pb_close()
        self.connected = False

    def is_connected(self) -> bool:
        try:
            return False if pb.pb_count_boards() < 0 else True
        except:
            return False

    def start(self):
        pb.pb_start()

    def reset(self):
        pb.pb_reset()

    def stop(self):
        pb.pb_stop()

    # def set_clock(self, clock):
    #     pb.pb_core_clock(clock)

    def is_finished(self):
        # IDK do we check for 'waiting'??
        return pb.pb_read_status() == 0

    def get_status(self):
        """
        See https://www.spincore.com/support/spinapi/reference/production/2013-09-25/spinapi_8h.html#ade910c40db242fb8238d29462d46de78
        Read status from the board. Not all boards support this, see your manual. Each bit of the
        returned integer indicates whether the board is in that state. Bit 0 is the least
        significant bit.

        Bit 0 - Stopped
        Bit 1 - Reset
        Bit 2 - Running
        Bit 3 - Waiting
        Bit 4 - Scanning (RadioProcessor boards only)

        *Note on Reset Bit: The Reset Bit will be true as soon as the board is initialized. *It
        will remain true until a hardware or software trigger occurs, *at which point it will stay
        false until the board is reset again.

        *Note on Activation Levels: The activation level of each bit depends on the board, please
        see *your product's manual for details.

        Bits 5-31 are reserved for future use.
        It should not be assumed that these will be set to 0.
        """
        return pb.pb_read_status()

    ##############################################
    # Functions for sequences
    ##############################################

    # TODO ideally sequences would have CONST chs.
    def get_available_sequences(self):
        sequences = {
            "MockSGAndorCWESR": seq_cw_esr,
            "SGAndorCWESR": seq_cw_esr,
            "SGAndorCWESRLongExposure": seq_cw_esr_long_exposure,
            "SGAndorPESR": seq_p_esr,
            "SGAndorRabi": seq_rabi,
            "SGAndorT1": seq_t1,
            "SGAndorRamsey": seq_ramsey,
            "SGAndorSpinEcho": seq_spin_echo,
        }
        return sequences

    def load_seq(self, seq_name, **seq_kwargs):
        logger.info("Loading {} sequence", seq_name)
        sequences = self.get_available_sequences()
        sequences[seq_name](self, self.sequence_params, **seq_kwargs)
        logger.info("Loaded {} sequence", seq_name)

    ##############################################
    # Functions for programming the pulseblaster
    ##############################################

    def start_programming(self):
        pb.pb_start_programming(pb.PULSE_PROGRAM)

    def stop_programming(self):
        pb.pb_stop_programming()

    def inst_pbonly(self, chs, opcode, data, duration):
        return pb.pb_inst_pbonly(chs, opcode, data, duration)

    def get_chs_cmd_bits(self, ch_defs, ch_list):
        command_num = 0
        for ch in ch_list:
            command_num = command_num + int(ch_defs[ch], 2)
        return command_num

    def end_sequence(self, dur):
        pb.pb_inst_pbonly(0, Inst.STOP, 0, dur)

    def add_instruction(
        self,
        active_chs,
        dur=0,
        delay=None,
        loop=None,
        num=0,
        inst=None,
        const_chs=(),
        **kwargs,
    ):
        """
        Function to add an arbitary pulse to the pulseblaster
        inputs:
            active_chs: list of pulse chs
            pulse_dur: duration of the pulse in ns
            pulse_delay: delay of the pulse in ns
            loop: parameter for loop control, "start", "end", None
            num: Number of loops
            inst: instruction to the start of a pulseblaster loop
            const_trigger: list of chs that are constant during the pulse

        output:
            Returns the instruction to the start of the pulseblaster loop
        """

        inst_out = None
        # check if a constant trigger is needed.
        # For some trigger modes the camera trigger is on during the whole intergration
        if len(const_chs) > 0:
            # append the constant trigger to the pulse chs list
            active_chs.append(const_chs[0])

        # Check if the pulse is long enough to be added
        if dur < self.shortest_dur:
            old_dur = dur
            if dur == 0:
                dur = 0
            else:
                dur = self.shortest_dur
            logger.error(
                "Pulse with channels {} rounded from {} ns to {} ns.",
                active_chs,
                old_dur,
                dur,
            )

        if dur >= self.shortest_dur:
            # make sure that that the pulse duration is a multiple of 2
            dur = np.round(dur / 2) * 2

            # define the pulse
            ctl = self.get_chs_cmd_bits(self.ch_defs, active_chs)

            # add the pulse to the pulseblaster with the three options
            # Starting a pulseblaster loop
            # Continuing
            # Ending a pulseblaster loop
            if loop == "start":
                inst_out = self.inst_pbonly(ctl, Inst.LOOP, num, dur)
            elif loop == "end":
                inst_out = self.inst_pbonly(ctl, Inst.END_LOOP, inst, dur)
            else:
                inst_out = self.inst_pbonly(ctl, Inst.CONTINUE, 0, dur)
        else:
            logger.error("Pulse duration too short, not adding pulse.")
            logger.error("Did not add {} for {} ns.", active_chs, dur)

        return inst_out

    def add_kernel(self, pulse_kernel, num_loop, const_chs=[], **kwargs):
        """
        Add the instructions to the signal generator assuming that the instructions that have been defined is a kernel
        """
        pulse_kernel.convert_to_instructions(const_chs=const_chs)
        # add the instructions to the pulseblaster in a loop
        for inst in pulse_kernel.insts:
            # For kernels that contain a single instruction no loop is required.
            if len(pulse_kernel.insts) == 1:
                self.add_instruction(**inst)
                return
            # For the first instruction, we need to add the start loop instruction
            if pulse_kernel.insts.index(inst) == 0:
                loop_inst = self.add_instruction(loop="start", num=num_loop, **inst)
            # For the last instruction, we need to add the end loop instruction
            elif pulse_kernel.insts.index(inst) == len(pulse_kernel.insts) - 1:
                self.add_instruction(loop="end", inst=loop_inst, **inst)
            else:
                self.add_instruction(**inst)
        return

    ##############################################

    def laser_output(self, onoff: bool):
        if onoff:
            chs = self.ch_defs
            control = self.get_chs_cmd_bits(chs, ["laser"])
            self.start_programming()
            self.inst_pbonly(control, Inst.BRANCH, 0, 1000 * Timing.ms)
            self.stop_programming()
            self.reset()
            self.start()
            logger.info("Laser set to on")
        else:
            self.stop()
            logger.info("Laser set to off")

    def rf_output(self, onoff: bool):
        if onoff:
            chs = self.ch_defs
            control = self.get_chs_cmd_bits(chs, ["mw_x"])
            self.start_programming()
            self.inst_pbonly(control, Inst.BRANCH, 0, 1000 * Timing.ms)
            self.stop_programming()
            self.reset()
            self.start()
            logger.info("RF set to on")
        else:
            self.reset()
            self.stop()
            logger.info("RF set to off")

    def laser_rf_output(self, onoff: bool):
        if onoff:
            chs = self.ch_defs
            control = self.get_chs_cmd_bits(chs, ["laser", "mw_x"])
            self.start_programming()
            self.inst_pbonly(control, Inst.BRANCH, 0, 1000 * Timing.ms)
            self.stop_programming()
            self.reset()
            self.start()
            logger.info("Laser and RF set to on")
        else:
            self.stop()
            logger.info("Laser and RF set to off")

    def output_on(self, state, ch_names):
        if state:
            chs = self.ch_defs
            control = self.get_chs_cmd_bits(chs, ch_names)
            self.start_programming()
            self.inst_pbonly(control, Inst.BRANCH, 0, 1000 * Timing.ms)
            self.stop_programming()

            self.reset()
            self.start()
        else:
            self.stop()
