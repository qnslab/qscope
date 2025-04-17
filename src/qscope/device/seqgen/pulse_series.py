import matplotlib.pyplot as plt
import numpy as np


class PulseSeries:
    def __init__(self, all_flags):
        # all_flags is a list of all the flags
        self.all_flags = all_flags
        # List of all of the pulses that have been added
        self.all_pulses = {
            flag: {"start": [], "dur": [], "state": []} for flag in all_flags
        }

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
        self.plt_hieight = 0.9

        self.define_shortest_instruction_length()
        self.pulses_shifted = False

    def define_shortest_instruction_length(self, val=12e-9):
        self.shortest_inst_len = val

    def add_pulse(self, flags, start_time, dur):
        if flags is None or flags == [] or flags == "":
            # if no flag is given, add the pulse to all flags with the set low state
            for flag in self.all_flags:
                self.all_pulses[flag]["start"].append(start_time)
                self.all_pulses[flag]["dur"].append(dur)
                self.all_pulses[flag]["state"].append(False)
        else:
            for flag in flags:
                self.all_pulses[flag]["start"].append(start_time)
                self.all_pulses[flag]["dur"].append(dur)
                self.all_pulses[flag]["state"].append(True)

        self.all_pulses
        return start_time + dur

    def add_delay(self, start_time, delay):
        for flag in self.all_flags:
            self.all_pulses[flag]["start"].append(start_time)
            self.all_pulses[flag]["dur"].append(delay)
            self.all_pulses[flag]["state"].append(False)
        return start_time + delay

    def get_total_time(self):
        self.total_time = 0
        total_time = 0

        # Get all of the end times of the pulses
        end_times = []
        for key in self.all_pulses:
            for idx in range(len(self.all_pulses[key]["start"])):
                end_times.append(
                    self.all_pulses[key]["start"][idx]
                    + self.all_pulses[key]["dur"][idx]
                )
        total_time = np.max(end_times)
        return total_time

    def shift_pulses(self):
        # check if any of the pulses start at a negative time.
        # If so, remove the negative time and adjust the pulse to account for this at the end of the sequence.
        # FIXME This function need to deal with some edge cases with multiple pulses with negative start times that don't have the same start time

        if self.pulses_shifted:
            ValueError("Pulses have already been shifted")
            return
        prev_end_time = self.get_total_time()
        for key in self.all_pulses:
            if len(self.all_pulses[key]["start"]) > 0:
                if min(self.all_pulses[key]["start"]) < 0:
                    shift_time = abs(min(self.all_pulses[key]["start"]))
                    self.all_pulses[key]["start"][0] = 0
                    self.all_pulses[key]["start"][0] -= shift_time
                    # Add shift time as a pulse at the end of the sequence
                    self.add_pulse([key], prev_end_time, shift_time)

        self.pulses_shifted = True

    def convert_to_instructions(self, const_flags=[]):
        # Define the instruction list
        insts = []
        # Collect all unique start and end times
        unique_times = set()
        for flag in self.all_pulses:
            unique_times.update(self.all_pulses[flag]["start"])
            unique_times.update(
                self.all_pulses[flag]["start"][idx] + self.all_pulses[flag]["dur"][idx]
                for idx in range(len(self.all_pulses[flag]["start"]))
            )

        unique_times = sorted(unique_times)

        # Create instructions for each unique time
        for time in unique_times:
            active_flags = []
            for flag in self.all_pulses:
                for idx in range(len(self.all_pulses[flag]["start"])):
                    start_time = self.all_pulses[flag]["start"][idx]
                    end_time = start_time + self.all_pulses[flag]["dur"][idx]
                    if start_time <= time < end_time:
                        # check if the flag is active at the current time
                        if self.all_pulses[flag]["state"][idx]:
                            active_flags.append(flag)

            insts.append({"time": time, "active_flags": active_flags})

        # define the duration of each instruction as the time difference between the current time and the next time
        for idx in range(len(insts) - 1):
            insts[idx]["dur"] = insts[idx + 1]["time"] - insts[idx]["time"]
        # The last instruction will have a duration of 0 so we will remove it
        insts = insts[:-1]

        # Add the constant flags to all of the instructions
        updated_insts = []
        for inst in insts:
            updated_insts.append(
                {
                    "active_flags": inst["active_flags"],
                    "dur": inst["dur"],
                    "const_flags": const_flags,
                }
            )

        self.insts = updated_insts
        return

    def add_kernal_instructions(self, siggen, num_loops):
        """
        Add the instructions to the signal generator assuming that the instructions that have been defined is a kernel
        """
        # add the instructions to the pulseblaster in a loop
        for inst in self.insts:
            # For the first instruction, we need to add the start loop instruction
            if self.insts.index(inst) == 0:
                loop_inst = siggen.add_instruction(
                    inst, loop="start", num_loops=num_loops
                )
            # For the last instruction, we need to add the end loop instruction
            elif self.insts.index(inst) == len(self.insts) - 1:
                siggen.add_instruction(inst, loop="end", inst=loop_inst)
            else:
                siggen.add_instruction(inst)
        return

    # PLOTS

    def plot_pulses(self):
        self.get_total_time()
        flag_states = [0] * len(self.all_flags)
        for i in range(len(self.all_flags)):
            flag_states[i] = flag_states[i] + i * self.plt_offset

        for flag in self.all_pulses:
            flag_index = list(self.all_pulses.keys()).index(flag)
            if len(self.all_pulses[flag]["start"]) > 0:
                for idx in range(len(self.all_pulses[flag]["start"])):
                    t = self.all_pulses[flag]["start"][idx]
                    dur = self.all_pulses[flag]["dur"][idx]
                    if self.all_pulses[flag]["state"][idx]:
                        plt.plot(
                            [t, t + dur],
                            [
                                flag_states[flag_index] + self.plt_hieight,
                                flag_states[flag_index] + self.plt_hieight,
                            ],
                            color=self.line_colors[flag_index],
                        )
                        plt.fill_between(
                            [t, t + dur],
                            flag_states[flag_index],
                            flag_states[flag_index] + self.plt_hieight,
                            color=self.fill_colors[flag_index],
                        )
                        plt.plot(
                            [t, t],
                            [
                                flag_states[flag_index],
                                flag_states[flag_index] + self.plt_hieight,
                            ],
                            self.line_colors[flag_index],
                        )
                        plt.plot(
                            [t + dur, t + dur],
                            [
                                flag_states[flag_index],
                                flag_states[flag_index] + self.plt_hieight,
                            ],
                            self.line_colors[flag_index],
                        )
                    else:
                        plt.plot(
                            [t, t + dur],
                            [flag_states[flag_index], flag_states[flag_index]],
                            self.line_colors[flag_index],
                        )
                if (
                    self.all_pulses[flag]["start"][-1]
                    + self.all_pulses[flag]["dur"][-1]
                    < self.total_time
                ):
                    plt.plot(
                        [
                            self.all_pulses[flag]["start"][-1]
                            + self.all_pulses[flag]["dur"][-1],
                            self.total_time,
                        ],
                        [flag_states[flag_index], flag_states[flag_index]],
                        self.line_colors[flag_index],
                    )
            else:
                plt.plot(
                    [0, self.total_time],
                    [flag_states[flag_index], flag_states[flag_index]],
                    self.line_colors[flag_index],
                )

        print(flag_states)
        plt.yticks(
            np.linspace(0, len(flag_states), len(flag_states)),
            list(self.all_flags.keys()),
        )
        plt.xlabel("Time (s)")
        plt.show()

    def plot_inst_kernel(self):
        insts = self.insts
        all_flags = self.all_flags

        plt.figure(figsize=(10, 5))

        # estimate the total time of the sequence
        total_time = 0
        for inst in insts:
            total_time += inst["dur"]

        t = 0
        flags = np.zeros(8)
        # make sure each flag has a unique y value
        for i in range(8):
            flags[i] = flags[i] + i * 1.1

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

        prev_flags = []
        prev_const_flags = []

        for inst in insts:
            # iterate over the flags and set the flags array to the correct value
            for flag in inst["active_flags"]:
                # get the index of the flag
                flag_index = list(all_flags.keys()).index(flag)
                # plot the time the flag is high
                plt.plot(
                    [t, t + inst["dur"]],
                    [flags[flag_index] + 1, flags[flag_index] + 1],
                    color=line_colors[flag_index],
                )
                # Plot a fill between the 0 and 1 of the flag
                plt.fill_between(
                    [t, t + inst["dur"]],
                    flags[flag_index],
                    flags[flag_index] + 1,
                    color=fill_colors[flag_index],
                )
                if flag not in prev_flags:
                    # Plot the step
                    plt.plot(
                        [t, t],
                        [flags[flag_index], flags[flag_index] + 1],
                        line_colors[flag_index],
                    )
                    # plot the time the flag is low
                    # plt.plot([t+inst["dur"], t+inst["dur"]], [flags[flag_index] + 1, flags[flag_index]], line_colors[flag_index])

            for flag in inst["const_flags"]:
                # get the index of the flag
                flag_index = list(all_flags.keys()).index(flag)
                dur = inst["dur"]
                # plot the time the flag is high
                plt.plot(
                    [t, t + dur],
                    [flags[flag_index] + 1, flags[flag_index] + 1],
                    line_colors[flag_index],
                )

                # Plot a fill between the 0 and 1 of the flag
                plt.fill_between(
                    [t, t + dur],
                    flags[flag_index],
                    flags[flag_index] + 1,
                    color=fill_colors[flag_index],
                )

                if flag not in prev_const_flags:
                    # Plot the step on
                    plt.plot(
                        [t, t],
                        [flags[flag_index], flags[flag_index] + 1],
                        line_colors[flag_index],
                    )

            # plot all zeros for the inst delay except for const flags
            for flag in all_flags:
                if flag not in inst["const_flags"]:
                    flag_index = list(all_flags.keys()).index(flag)
                    plt.plot(
                        [t + inst["dur"], t + inst["dur"]],
                        [flags[flag_index], flags[flag_index]],
                        line_colors[flag_index],
                    )
                if flag in prev_const_flags and flag not in inst["const_flags"]:
                    flag_index = list(all_flags.keys()).index(flag)
                    # Plot step off
                    plt.plot(
                        [t, t],
                        [flags[flag_index] + 1, flags[flag_index]],
                        line_colors[flag_index],
                    )

            # plot the constant 0 of the other flags that are not on
            for flag in all_flags:
                if flag not in inst["active_flags"] and flag not in inst["const_flags"]:
                    flag_index = list(all_flags.keys()).index(flag)
                    plt.plot(
                        [t, t + inst["dur"]],
                        [flags[flag_index], flags[flag_index]],
                        line_colors[flag_index],
                    )
                if flag in prev_flags and flag not in inst["active_flags"]:
                    flag_index = list(all_flags.keys()).index(flag)
                    # Plot step off
                    plt.plot(
                        [t, t],
                        [flags[flag_index] + 1, flags[flag_index]],
                        line_colors[flag_index],
                    )

            prev_flags = inst["active_flags"]

            prev_const_flags = inst["const_flags"]
            t += inst["dur"]

        plt.yticks(flags + 0.5, list(all_flags.keys()))
        plt.xlabel("Time (s)")
        plt.show()
