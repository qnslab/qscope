import copy

import matplotlib.pyplot as plt
import numpy as np
from PyQt6.QtWidgets import QLabel, QSpinBox, QVBoxLayout, QWidget


class PulseKernel:
    def __init__(self, ch_defs):
        # ch_defs is a list of all the chs
        self.ch_defs = ch_defs
        # List of all of the pulses that have been added
        self.kernel = {
            ch: {"start": [], "dur": [], "state": [], "var_dur": [], "ch_delay": []}
            for ch in ch_defs
        }

        # Parameters for plotting the pulses
        # Define a list of colors from tab20
        colormap = plt.get_cmap("tab20")
        # set every 2nd color for the line color
        self.fill_colors = [colormap(i) for i in range(1, 20, 2)]
        self.line_colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]

        self.plt_offset = 1
        self.plt_height = 0.9

        self.shortest_pulse = int(12)  # ns

    def finish_kernel(self):
        self.kernel_unaltered = copy.deepcopy(self.kernel)

    def check_duration(self, dur, var_dur):
        if var_dur:
            # Long enough to ensure a pulse is generated
            # The actual duration will be overwritten by the update_var_durs function
            dur = 12

        # FIXME add a proper catch for this error and handle it.
        if dur < self.shortest_pulse:
            if dur < self.shortest_pulse / 2:
                dur = 0
            else:
                dur = self.shortest_pulse

        return dur

    def append_pulse(self, chs, dur, var_dur=False, ch_delay=0):
        """
        Append a pulse to the end of the sequence
            ch: ch to add the pulse to
            dur (s): duration of the pulse
            state: state of the pulse
            ch_delay (s): delay time for the ch
        """
        dur = self.check_duration(dur, var_dur)

        start_time = self.get_end_time()
        for ch in chs:
            self.kernel[ch]["start"].append(start_time)
            self.kernel[ch]["dur"].append(dur)
            self.kernel[ch]["state"].append(True)
            self.kernel[ch]["var_dur"].append(var_dur)
            self.kernel[ch]["ch_delay"].append(ch_delay)
        return

    def add_pulse(self, chs, start_time, dur, var_dur=False, ch_delay=0):
        """
        Add a pulse to the sequence
            chs: list of chs to add the pulse to
            start_time (s): start time of the pulse
            dur (s): duration of the pulse
            ch_delay (s): delay time for the ch
        """

        dur = self.check_duration(dur, var_dur)

        if chs is None or chs == [] or chs == "":
            # if no ch is given, add the pulse to all chs with the set low state
            for ch in self.ch_defs:
                self.kernel[ch]["start"].append(start_time)
                self.kernel[ch]["dur"].append(dur)
                self.kernel[ch]["ch_delay"].append(ch_delay)
                self.kernel[ch]["var_dur"].append(var_dur)
                self.kernel[ch]["state"].append(False)
        else:
            for ch in chs:
                self.kernel[ch]["start"].append(start_time)
                self.kernel[ch]["dur"].append(dur)
                self.kernel[ch]["ch_delay"].append(ch_delay)
                self.kernel[ch]["var_dur"].append(var_dur)
                self.kernel[ch]["state"].append(True)

        self.kernel
        return start_time + dur

    def append_delay(
        self,
        dur,
        var_dur=False,
    ):
        """
        Append a delay to the end of the sequence
        """
        if dur > 0:
            dur = self.check_duration(dur, var_dur)
            start_time = self.get_end_time()
            # Make sure the start time is devisable by 2
            start_time = start_time
            for ch in self.ch_defs:
                self.kernel[ch]["start"].append(start_time)
                self.kernel[ch]["dur"].append(dur)
                self.kernel[ch]["ch_delay"].append(0)
                self.kernel[ch]["var_dur"].append(var_dur)
                self.kernel[ch]["state"].append(False)
            return

    def add_delay(self, start_time, dur, var_dur=False, kernel=None):
        """
        Add a delay to all of the chs
        """
        if kernel is None:
            kernel = self.kernel

        dur = self.check_duration(dur, var_dur)

        for ch in self.ch_defs:
            kernel[ch]["start"].append(start_time)
            kernel[ch]["dur"].append(dur)
            kernel[ch]["ch_delay"].append(0)
            kernel[ch]["var_dur"].append(var_dur)
            kernel[ch]["state"].append(False)
        return kernel

    def print_pulses(self):
        for ch in self.kernel:
            print(ch)
            for idx in range(len(self.kernel[ch]["start"])):
                # if self.kernel[ch]['state'][idx]:
                print(
                    f"Start: {self.kernel[ch]['start'][idx]:0.3e}, "
                    + f"Dur: {self.kernel[ch]['dur'][idx]}, "
                    + f"Ch delay: {self.kernel[ch]['ch_delay'][idx]}, "
                    + f"Var dur: {self.kernel[ch]['var_dur'][idx]}, "
                    + f"state: {self.kernel[ch]['state'][idx]}"
                )
        return

    def reset_kernel(self):
        self.kernel = copy.deepcopy(self.kernel_unaltered)

    def update_var_durs(self, new_dur):
        """
        Update the variable durations of the pulses and shift the start times
        that are after that pulse accordingly.
        """
        # get a list of all the pulses that have a variable durati
        # on

        # reset the kernel to the unaltered kernel
        self.reset_kernel()
        kernel = self.kernel

        var_dur_pulses = []
        for ch in kernel:
            for idx in range(len(kernel[ch]["start"])):
                if self.kernel[ch]["var_dur"][idx]:
                    var_dur_pulses.append(
                        (ch, idx, kernel[ch]["start"][idx], kernel[ch]["dur"][idx])
                    )

        # sort the pulses by the start time
        var_dur_pulses = sorted(var_dur_pulses, key=lambda x: x[2])

        # check for var_dur_pulses that have the same start time and duration and remove all but one
        var_dur_pulses = [
            var_dur_pulses[i]
            for i in range(len(var_dur_pulses))
            if i == 0 or var_dur_pulses[i][2] != var_dur_pulses[i - 1][2]
        ]

        pulse_idx = 0
        # update the durations of the pulses
        for ch, idx, start_time, dur in var_dur_pulses:
            orig_dur = kernel[ch]["dur"][idx]

            # update the start time for the subsequent variable duration pulses as they will be shifted
            if pulse_idx > 0:
                start_time += new_dur - orig_dur

            # Iterate over all of the pulses and shift the start times of the pulses that are after the current pulse in this channel
            for ch2 in self.kernel:
                for idx2 in range(len(self.kernel[ch2]["start"])):
                    start_time_2 = self.kernel[ch2]["start"][idx2]
                    if start_time_2 > start_time:
                        self.kernel[ch2]["start"][idx2] += new_dur - orig_dur

            # update the duration of the pulse
            self.kernel[ch]["dur"][idx] = new_dur

            pulse_idx += 1

        self.kernel = kernel

    def get_end_time(self):
        self.total_time = 0
        total_time = 0
        # Get all of the end times of the pulses
        end_times = []
        for key in self.kernel:
            for idx in range(len(self.kernel[key]["start"])):
                end_times.append(
                    self.kernel[key]["start"][idx] + self.kernel[key]["dur"][idx]
                )
        if len(end_times) > 0:
            self.total_time = np.max(end_times)
        else:
            self.total_time = 0
        return self.total_time

    def shift_ch_delays(self):
        """
        Shift each of the channels by thier delay time.
        Then set the delay time to 0 so it can't be shifted again
        Finally, check if the end time has changed and add a delay to the end of the sequence if it has
        """

        # subtract the ch_delay from the start time for all pulses
        # get to current total time for the sequence
        prev_end_time = self.get_end_time()

        kernel = self.kernel
        for key in kernel:
            if len(kernel[key]["start"]) > 0:
                for idx in range(0, len(kernel[key]["start"])):
                    if kernel[key]["state"] and kernel[key]["ch_delay"][idx] > 0:
                        kernel[key]["start"][idx] -= kernel[key]["ch_delay"][idx]
                        # Now set the ch_delay to 0 so it can't be subtracted again
                        kernel[key]["ch_delay"][idx] = 0

        # # Check if the total time has changed
        if prev_end_time != self.get_end_time():
            # then one of the pulses was defining the end of the sequence so we need to add a delay to the end of the sequence to make sure the sequence is the same length
            dur = abs(prev_end_time - self.get_end_time())
            self.add_delay(prev_end_time, dur, kernel)

        self.kernel = kernel

    def wrap_pulses(self):
        """
        Adjusts the start times of pulses in the sequence to ensure no pulse starts at a negative time.
        This function checks if any pulses in the sequence have a negative start time. If so, it adjusts the start time to zero and modifies the duration accordingly.
        The negative time is then added as a pulse at the end of the sequence. This function also ensures that pulses are only shifted once by checking the `pulses_shifted` attribute.
        Raises:
            ValueError: If pulses have already been shifted.
        """

        # Now shift the negative time pulses to the end.
        prev_end_time = self.get_end_time()

        for key in self.kernel:
            # Find all pulses with negative start times
            negative_pulses = [
                (idx, start_time)
                for idx, start_time in enumerate(self.kernel[key]["start"])
                if start_time < 0
            ]

            for idx, start_time in negative_pulses:
                shift_time = abs(start_time)
                dur = self.kernel[key]["dur"][idx]

                # Determine the duration of the shifted pulse at the end of the sequence
                if dur > shift_time:
                    dur_end = shift_time
                else:
                    dur_end = dur

                # Add the shifted pulse to the end of the sequence
                self.add_pulse([key], prev_end_time + start_time, dur_end, 0)

                # If the pulse duration is longer than the negative start time, add the remaining part of the pulse
                if self.kernel[key]["dur"][idx] > shift_time:
                    self.add_pulse(
                        [key], 0, self.kernel[key]["dur"][idx] - shift_time, 0
                    )

        # Finally, remove any pulses that have a start time less than 0
        for key in self.kernel:
            negative_pulses = [
                (idx, start_time)
                for idx, start_time in enumerate(self.kernel[key]["start"])
                if start_time < 0
            ]
            # Remove the negative pulses going backwards so that the indexes don't change
            for idx, start_time in negative_pulses[::-1]:
                self.kernel[key]["start"].pop(idx)
                self.kernel[key]["dur"].pop(idx)
                self.kernel[key]["ch_delay"].pop(idx)
                self.kernel[key]["state"].pop(idx)
                self.kernel[key]["var_dur"].pop(idx)
        self.pulses_shifted = True

    def convert_to_instructions(self, const_chs=[]):
        # shift the channels
        self.shift_ch_delays()
        # wrap the pulses
        self.wrap_pulses()

        # Define the instruction list
        insts = []
        # Collect all unique start and end times
        unique_times = set()
        for ch in self.kernel:
            unique_times.update(self.kernel[ch]["start"])
            unique_times.update(
                self.kernel[ch]["start"][idx] + self.kernel[ch]["dur"][idx]
                for idx in range(len(self.kernel[ch]["start"]))
            )

        unique_times = sorted(unique_times)

        # Create instructions for each unique time
        for time in unique_times:
            active_chs = []
            for ch in self.kernel:
                for idx in range(len(self.kernel[ch]["start"])):
                    start_time = self.kernel[ch]["start"][idx]
                    end_time = start_time + self.kernel[ch]["dur"][idx]
                    if start_time <= time < end_time:
                        # check if the ch is active at the current time
                        if self.kernel[ch]["state"][idx]:
                            active_chs.append(ch)

            insts.append({"time": time, "active_chs": active_chs})

        # define the duration of each instruction as the time difference between the current time and the next time
        for idx in range(len(insts) - 1):
            insts[idx]["dur"] = insts[idx + 1]["time"] - insts[idx]["time"]
        # The last instruction will have a duration of 0 so we will remove it
        insts = insts[:-1]

        # Add the constant chs to all of the instructions
        updated_insts = []
        for inst in insts:
            updated_insts.append(
                {
                    "active_chs": inst["active_chs"],
                    "dur": inst["dur"],
                    "const_chs": const_chs,
                }
            )

        self.insts = updated_insts

        # Check for any instructions that are sequential and indentical in active chs
        # and combine them
        combined_insts = []
        for idx in range(len(self.insts) - 1):
            if self.insts[idx]["active_chs"] == self.insts[idx + 1]["active_chs"]:
                self.insts[idx + 1]["dur"] += self.insts[idx]["dur"]
            else:
                combined_insts.append(self.insts[idx])

        # Add the last instruction
        combined_insts.append(self.insts[-1])
        self.insts = combined_insts
        return

    def add_kernal_instructions(self, seqgen, num_loops, const_chs=[]):
        """
        Add the instructions to the signal generator assuming that the instructions that have been defined is a kernel
        """
        self.convert_to_instructions(const_chs=const_chs)
        # add the instructions to the pulseblaster in a loop
        for inst in self.insts:
            # For the first instruction, we need to add the start loop instruction
            if self.insts.index(inst) == 0:
                loop_inst = seqgen.add_instruction(
                    inst, loop="start", num_loops=num_loops
                )
            # For the last instruction, we need to add the end loop instruction
            elif self.insts.index(inst) == len(self.insts) - 1:
                seqgen.add_instruction(inst, loop="end", inst=loop_inst)
            else:
                seqgen.add_instruction(inst)
        return

    # PLOTS

    def plot_pulses(self):
        self.get_end_time()

        plt.figure(figsize=(10, 5))
        ch_states = [0] * len(self.ch_defs)
        for i in range(len(self.ch_defs)):
            ch_states[i] = ch_states[i] + i * self.plt_offset

        for ch in self.kernel:
            ch_index = list(self.kernel.keys()).index(ch)
            if len(self.kernel[ch]["start"]) > 0:
                for idx in range(len(self.kernel[ch]["start"])):
                    t = self.kernel[ch]["start"][idx]
                    dur = self.kernel[ch]["dur"][idx]
                    if self.kernel[ch]["state"][idx]:
                        plt.plot(
                            [t, t + dur],
                            [
                                ch_states[ch_index] + self.plt_height,
                                ch_states[ch_index] + self.plt_height,
                            ],
                            color=self.line_colors[ch_index],
                        )
                        plt.fill_between(
                            [t, t + dur],
                            ch_states[ch_index],
                            ch_states[ch_index] + self.plt_height,
                            color=self.fill_colors[ch_index],
                        )
                        plt.plot(
                            [t, t],
                            [
                                ch_states[ch_index],
                                ch_states[ch_index] + self.plt_height,
                            ],
                            self.line_colors[ch_index],
                        )
                        plt.plot(
                            [t + dur, t + dur],
                            [
                                ch_states[ch_index],
                                ch_states[ch_index] + self.plt_height,
                            ],
                            self.line_colors[ch_index],
                        )
                    else:
                        plt.plot(
                            [t, t + dur],
                            [ch_states[ch_index], ch_states[ch_index]],
                            self.line_colors[ch_index],
                        )
                if (
                    self.kernel[ch]["start"][-1] + self.kernel[ch]["dur"][-1]
                    < self.total_time
                ):
                    plt.plot(
                        [
                            self.kernel[ch]["start"][-1] + self.kernel[ch]["dur"][-1],
                            self.total_time,
                        ],
                        [ch_states[ch_index], ch_states[ch_index]],
                        self.line_colors[ch_index],
                    )
            else:
                plt.plot(
                    [0, self.total_time],
                    [ch_states[ch_index], ch_states[ch_index]],
                    self.line_colors[ch_index],
                )

        plt.yticks(
            np.linspace(0, len(ch_states), len(ch_states)), list(self.ch_defs.keys())
        )
        plt.xlabel("Time (s)")
        plt.show()

    def plot_inst_kernel(self):
        insts = self.insts
        ch_defs = self.ch_defs

        plt.figure(figsize=(10, 5))

        # estimate the total time of the sequence
        total_time = 0
        for inst in insts:
            total_time += inst["dur"]

        t = 0
        chs = np.zeros(len(ch_defs))
        num_chs = len(ch_defs)
        # make sure each ch has a unique y value
        for i in range(num_chs):
            chs[i] = chs[i] + i * 1.1

        # Define a list of colors from tab20
        colormap = plt.get_cmap("tab20")
        # set every 2nd color for the line color
        line_colors = [colormap(i) for i in range(0, 20, 2)]
        fill_colors = [colormap(i) for i in range(1, 20, 2)]
        line_colors = [
            "tab:blue",
            "tab:orange",
            "tab:green",
            "tab:red",
            "tab:purple",
            "tab:brown",
            "tab:pink",
            "tab:gray",
            "tab:olive",
            "tab:cyan",
        ]

        prev_chs = []
        prev_const_chs = []

        for inst in insts:
            # iterate over the chs and set the chs array to the correct value
            for ch in inst["active_chs"]:
                # get the index of the ch
                ch_index = list(ch_defs.keys()).index(ch)
                # plot the time the ch is high
                plt.plot(
                    [t, t + inst["dur"]],
                    [chs[ch_index] + 1, chs[ch_index] + 1],
                    color=line_colors[ch_index],
                )
                # Plot a fill between the 0 and 1 of the ch
                plt.fill_between(
                    [t, t + inst["dur"]],
                    chs[ch_index],
                    chs[ch_index] + 1,
                    color=fill_colors[ch_index],
                )
                if ch not in prev_chs:
                    # Plot the step
                    plt.plot(
                        [t, t],
                        [chs[ch_index], chs[ch_index] + 1],
                        line_colors[ch_index],
                    )
                    # plot the time the ch is low
                    # plt.plot([t+inst["dur"], t+inst["dur"]], [chs[ch_index] + 1, chs[ch_index]], line_colors[ch_index])

            for ch in inst["const_chs"]:
                # get the index of the ch
                ch_index = list(ch_defs.keys()).index(ch)
                dur = inst["dur"]
                # plot the time the ch is high
                plt.plot(
                    [t, t + dur],
                    [chs[ch_index] + 1, chs[ch_index] + 1],
                    line_colors[ch_index],
                )

                # Plot a fill between the 0 and 1 of the ch
                plt.fill_between(
                    [t, t + dur],
                    chs[ch_index],
                    chs[ch_index] + 1,
                    color=fill_colors[ch_index],
                )

                if ch not in prev_const_chs:
                    # Plot the step on
                    plt.plot(
                        [t, t],
                        [chs[ch_index], chs[ch_index] + 1],
                        line_colors[ch_index],
                    )

            # plot all zeros for the inst delay except for const chs
            for ch in ch_defs:
                if ch not in inst["const_chs"]:
                    ch_index = list(ch_defs.keys()).index(ch)
                    plt.plot(
                        [t + inst["dur"], t + inst["dur"]],
                        [chs[ch_index], chs[ch_index]],
                        line_colors[ch_index],
                    )
                if ch in prev_const_chs and ch not in inst["const_chs"]:
                    ch_index = list(ch_defs.keys()).index(ch)
                    # Plot step off
                    plt.plot(
                        [t, t],
                        [chs[ch_index] + 1, chs[ch_index]],
                        line_colors[ch_index],
                    )

            # plot the constant 0 of the other chs that are not on
            for ch in ch_defs:
                if ch not in inst["active_chs"] and ch not in inst["const_chs"]:
                    ch_index = list(ch_defs.keys()).index(ch)
                    plt.plot(
                        [t, t + inst["dur"]],
                        [chs[ch_index], chs[ch_index]],
                        line_colors[ch_index],
                    )
                if ch in prev_chs and ch not in inst["active_chs"]:
                    ch_index = list(ch_defs.keys()).index(ch)
                    # Plot step off
                    plt.plot(
                        [t, t],
                        [chs[ch_index] + 1, chs[ch_index]],
                        line_colors[ch_index],
                    )

            prev_chs = inst["active_chs"]

            prev_const_chs = inst["const_chs"]
            t += inst["dur"]

        plt.yticks(chs + 0.5, list(ch_defs.keys()))
        plt.xlabel("Time (s)")
        # plt.show()
