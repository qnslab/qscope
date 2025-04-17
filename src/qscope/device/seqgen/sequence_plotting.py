import numpy as np
from matplotlib import pyplot as plt


def plot_instructions_kernel(insts, ch_defs):
    plt.figure(figsize=(10, 5))

    # estimate the total time of the sequence
    total_time = 0
    for inst in insts:
        total_time += inst["dur"] + inst["ch_delay"]

    t = 0
    chs = np.zeros(8)
    # make sure each ch has a unique y value
    for i in range(8):
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
    prev_delay = 0

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
                    [t, t], [chs[ch_index], chs[ch_index] + 1], line_colors[ch_index]
                )
                # plot the time the ch is low
                # plt.plot([t+inst["dur"], t+inst["dur"]], [chs[ch_index] + 1, chs[ch_index]], line_colors[ch_index])

        for ch in inst["const_chs"]:
            # get the index of the ch
            ch_index = list(ch_defs.keys()).index(ch)
            dur = inst["dur"] + inst["ch_delay"]
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
                    [t, t], [chs[ch_index], chs[ch_index] + 1], line_colors[ch_index]
                )

        # plot all zeros for the inst delay except for const chs
        for ch in ch_defs:
            if ch not in inst["const_chs"]:
                ch_index = list(ch_defs.keys()).index(ch)
                plt.plot(
                    [t + inst["dur"], t + inst["dur"] + inst["ch_delay"]],
                    [chs[ch_index], chs[ch_index]],
                    line_colors[ch_index],
                )
            if ch in prev_const_chs and ch not in inst["const_chs"]:
                ch_index = list(ch_defs.keys()).index(ch)
                # Plot step off
                plt.plot(
                    [t - prev_delay, t - prev_delay],
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
                    [t - prev_delay, t - prev_delay],
                    [chs[ch_index] + 1, chs[ch_index]],
                    line_colors[ch_index],
                )

        prev_chs = inst["active_chs"]
        prev_delay = inst["ch_delay"]
        prev_const_chs = inst["const_chs"]
        t += inst["dur"] + inst["ch_delay"]

    plt.yticks(chs + 0.5, list(ch_defs.keys()))
    plt.xlabel("Time (s)")
    plt.show()

    # Plot the instruction kernel


def plot_pulses_kernel(ch_defs, chs):
    plt.figure(figsize=(10, 5))

    # estimate the total time of the sequence
    total_time = 0
    for ch in ch_defs:
        if len(ch_defs[ch]["start"]) > 0:
            total_time = max(
                total_time, max(ch_defs[ch]["start"]) + max(ch_defs[ch]["dur"])
            )

    chs = np.zeros(8)
    # make sure each ch has a unique y value
    for i in range(8):
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

    for ch in ch_defs:
        ch_index = list(ch_defs.keys()).index(ch)
        # iterate over the chs and set the chs array to the correct value
        if len(ch_defs[ch]["start"]) > 0:
            for idx in range(len(ch_defs[ch]["start"])):
                t = ch_defs[ch]["start"][idx]
                dur = ch_defs[ch]["dur"][idx]
                if ch_defs[ch]["state"][idx]:
                    # plot the time the ch is high
                    plt.plot(
                        [t, t + dur],
                        [chs[ch_index] + 1, chs[ch_index] + 1],
                        color=line_colors[ch_index],
                    )
                    plt.fill_between(
                        [t, t + dur],
                        chs[ch_index],
                        chs[ch_index] + 1,
                        color=fill_colors[ch_index],
                    )
                    # on off components
                    plt.plot(
                        [t, t],
                        [chs[ch_index], chs[ch_index] + 1],
                        line_colors[ch_index],
                    )
                    plt.plot(
                        [t + dur, t + dur],
                        [chs[ch_index], chs[ch_index] + 1],
                        line_colors[ch_index],
                    )
                else:
                    # plot the time the ch is low
                    plt.plot(
                        [t, t + dur],
                        [chs[ch_index], chs[ch_index]],
                        line_colors[ch_index],
                    )
            # Now determine when the ch is off and plot the off component
            if ch_defs[ch]["start"][0] > 0:
                plt.plot(
                    [0, ch_defs[ch]["start"][0]],
                    [chs[ch_index], chs[ch_index]],
                    line_colors[ch_index],
                )
        else:
            # plot the time the ch is low the entire time
            plt.plot(
                [0, total_time], [chs[ch_index], chs[ch_index]], line_colors[ch_index]
            )

    # plot the constant 0 of the other chs that are not on

    plt.yticks(chs + 0.5, list(ch_defs.keys()))
    plt.xlabel("Time (s)")
    plt.show()
